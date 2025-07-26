# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import os, sys, warnings
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # 첫 flag는 Test 때는 False였지만 True로 하면 A100에서 학습 빨라짐.
torch.backends.cudnn.allow_tf32 = True
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')
import time, argparse
import wandb, ruamel.yaml as yaml, numpy as np
import torch.distributed as dist
from glob import glob
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from timm.scheduler import create_scheduler_v2 as create_scheduler
###
from utils.logger_func import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from utils.misc import str2bool, manage_checkpoints, load_model_state_dict
from utils.optim import param_groups_weight_decay
from utils.data import random_crop_arr, center_crop_arr
from modelling.tokenizer import VQ_models
from losses.vq_loss import VQLoss


#################################################################################
#                                  Training Loop                                #
#################################################################################

def build_parser():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    default_config = 'configs/tokenizer/maetok-b-64.yaml'
    add_arg("--config", type=str, default=default_config)                   # config prms를 정의한 config 파일 경로

    data_path = "ImageNet2012/train"
    VQchoice = list(VQ_models.keys())
    add_arg("--exp-index", type=str, default=None)                          # 실험 idx
    add_arg("--data-path", type=str, default=data_path)                     # training에 사용할 데이터 경로
    add_arg("--cloud-save-path", type=str)                                  # ckpt를 저장할 cloud 디스크 경로 (지정하지 않으면 local 저장)
    add_arg("--no-local-save", type=str2bool, default=False)                # 디스크 용량 제한으로 local 저장을 하지 않을 경우 True
    add_arg("--vq-model", type=str, choices=VQchoice, default="VQ-16")      # 사용할 VQ 모델의 이름
    add_arg("--vq-ckpt", type=str, default=None)                            # 학습을 재개할 VQ 모델의 checkpoint 경로
    add_arg("--finetune", type=str2bool, default=False)                     # 사전학습된 VQ 모델을 finetune할지 여부
    add_arg("--ema", type=str2bool, default=True)                           # EMA (Exponential Moving Average) training을 사용할지 여부
    add_arg("--codebook-size", type=int, default=16384)                     # VQ를 위한 codebook의 entry 개수
    add_arg("--codebook-embed-dim", type=int, default=8)                    # codebook의 embedding 차원 수
    add_arg("--codebook-l2-norm", type=str2bool, default=True)              # codebook entry에 L2 normalization 적용 여부
    add_arg("--codebook-weight", type=float, default=1.0)                   # codebook loss의 weight
    add_arg("--entropy-loss-ratio", type=float, default=0.0)                # codebook loss 내 entropy loss의 비율
    add_arg("--vq-loss-ratio", type=float, default=1.0)                     # codebook loss 내 VQ loss 항의 비율
    add_arg("--commit-loss-beta", type=float, default=0.25)                 # VQ commit loss의 beta 계수
    add_arg("--reconstruction-weight", type=float, default=1.0)             # 이미지 recon- loss의 weight
    add_arg("--reconstruction-loss", type=str, default='l2')                # 이미지 recon-에 사용할 loss 종류 (예: l2)
    add_arg("--kl-loss-weight", type=float, default=0.000001)               # KL loss의 weight
    add_arg("--tau", type=float, default=0.1)                               # temperature 계수 (예: softmax 등에서 사용)
    add_arg("--num-codebooks", type=int, default=1)                         # 병렬로 사용하는 codebook 개수
    
    add_arg("--perceptual-weight", type=float, default=1.0)                 # LPIPS perceptual loss의 가중치
    add_arg("--perceptual-loss", type=str, default='vgg', choices=['vgg', 'timm', 'tv'])    # LPIPS에서 사용할 perceptual loss의 종류
    add_arg("--perceptual-model", type=str, default='vgg')                  # LPIPS에서 사용하는 perceptual loss 모델 이름
    add_arg("--perceptual-dino-variants", type=str, default='depth12_no_train')             # LPIPS용 DINO 모델의 variant 이름
    add_arg("--perceptual-intermediate-loss", type=str2bool, default=False)                 # LPIPS의 중간 feature에서 perceptual loss를 계산할지 여부
    add_arg("--perceptual-logit-loss", type=str2bool, default=False)        # LPIPS의 logits에서 perceptual loss를 계산할지 여부
    add_arg("--perceptual-resize", type=str2bool, default=False)            # 입력 이미지를 resize한 후 LPIPS loss를 계산할지 여부
    add_arg("--perceptual-warmup", type=int, default=None)                  # perceptual loss를 적용하기 시작할 iteration (warmup)
    
    add_arg("--disc-weight", type=float, default=0.5)                       # GAN 학습 시 discriminator loss의 가중치
    add_arg("--disc-start", type=int, default=20000)                        # discriminator 학습을 시작할 iteration 수
    add_arg("--disc-dim", type=int, default=64)                             # discriminator의 기본 channel 수
    add_arg("--disc-type", type=str, choices=['patchgan', 'stylegan', 'maskbit', 'dino'], default='patchgan')   # 사용할 discriminator의 구조 타입
    add_arg("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge')           # discriminator에 사용할 loss 종류
    add_arg("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge')                       # GAN 학습 시 generator에 적용할 loss 종류
    add_arg("--lecam-loss-weight", type=float, default=None)                # LeCam regularization loss의 가중치
    add_arg("--use-diff-aug", type=str2bool, default=False)                 # DiffAugment를 사용할지 여부
    add_arg("--disc-cr-loss-weight", type=float, default=0.0)               # GAN 학습 시 discriminator의 consistency regularization loss 가중치
    add_arg("--disc-adaptive-weight", type=str2bool, default=False)         # discriminator loss의 adaptive weighting 적용 여부
    
    add_arg("--compile", type=str2bool, default=False)                      # model을 torch.compile로 최적화할지 여부
    add_arg("--dropout-p", type=float, default=0.0)                         # dropout 확률값 (dropout probability)
    add_arg("--results-dir", type=str, default="results_tokenizer_image")   # 결과 파일을 저장할 디렉토리 경로
    add_arg("--dataset", type=str, default='imagenet')                      # 사용할 dataset 이름
    add_arg("--image-size", type=int, choices=[256, 512], default=256)      # 입력 이미지 크기
    add_arg("--epochs", type=int, default=40)                               # 전체 학습 epoch 수
    add_arg("--optimizer", type=str, default='adam')                        # 사용할 optimizer 종류
    add_arg("--lr", type=float, default=1e-4)                               # 학습률 (learning rate)
    add_arg("--lr_warmup_epochs", type=int, default=1)                      # learning rate warmup을 적용할 epoch 수
    add_arg("--lr_scheduler", type=str, default='none')                     # 사용할 learning rate scheduler 종류
    add_arg("--weight-decay", type=float, default=5e-2)                     # weight decay 값 (정규화를 위한 L2 penalty 계수)
    add_arg("--beta1", type=float, default=0.9)                             # Adam optimizer의 beta1 계수
    add_arg("--beta2", type=float, default=0.95)                            # Adam optimizer의 beta2 계수
    add_arg("--max-grad-norm", default=1.0, type=float)                     # gradient clipping을 위한 최대 gradient norm
    
    add_arg("--global-batch-size", type=int, default=128)                   # 전체 GPU에서 사용할 global batch size
    add_arg("--global-seed", type=int, default=0)                           # 전체 실험 reproducibility를 위한 random seed
    add_arg("--num-workers", type=int, default=16)                          # DataLoader에서 사용할 subprocess 수
    add_arg("--log-every", type=int, default=100)                           # logging을 출력할 iteration 간격
    add_arg("--vis-every", type=int, default=5000)                          # 시각화 결과를 저장할 iteration 간격
    add_arg("--ckpt-every", type=int, default=5000)                         # checkpoint를 저장할 iteration 간격
    add_arg("--gradient-accumulation-steps", type=int, default=1)           # gradient accumulation에 사용할 step 수
    add_arg("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])  # mixed precision 학습에서 사용할 precision 종류

    add_arg("--enc-type", type=str, default="cnn")                          # encoder 구조 타입 (예: cnn, vit 등)
    add_arg("--dec-type", type=str, default="cnn")                          # decoder 구조 타입 (예: cnn, vit 등)
    add_arg("--num-latent-tokens", type=int, default=None)                  # 사용할 latent token의 개수 (None이면 default 방식)
    add_arg("--encoder-model", type=str, default='vit_tiny_patch14_dinov2.lvd142m')                     # 사용할 encoder 모델 이름  ##### vit_small_patch14_dinov2
    add_arg("--decoder-model", type=str, default='vit_tiny_patch14_dinov2.lvd142m')                     # 사용할 decoder 모델 이름  ##### vit_small_patch14_dinov2
    add_arg("--encoder-tuning-method", type=str, default='full', choices=['full', 'lora', 'frozen'])    # encoder tuning 방식 (full, LoRA, frozen 중 선택)
    add_arg("--decoder-tuning-method", type=str, default='full', choices=['full', 'lora', 'frozen'])    # decoder tuning 방식 (full, LoRA, frozen 중 선택)
    add_arg("--encoder-pretrained", type=str2bool, default=True)            # encoder에 pre-trained weight를 로드할지 여부
    add_arg("--decoder-pretrained", type=str2bool, default=False)           # decoder에 pre-trained weight를 로드할지 여부
    add_arg("--encoder-patch-size", type=int, default=16)                   # encoder에서 사용하는 patch size
    add_arg("--decoder-patch-size", type=int, default=16)                   # decoder에서 사용하는 patch size
    add_arg("--to-pixel", type=str, default="linear")                       # latent를 이미지 픽셀로 변환하는 방식 (예: linear 등)

    # repa
    add_arg("--repa", type=str2bool, default=False, help='use repa')
    add_arg('--repa-model', type=str, default='vit_base_patch16_224', help='repa model name')
    add_arg('--repa-patch-size', type=int, default=16, help='repa patch size')
    add_arg('--repa-proj-dim', type=int, default=1024, help='repa embed dim')
    add_arg('--repa-loss-weight', type=float, default=0.1, help='repa loss weight')
    add_arg('--repa-align', type=str, default='global', help='align repa feature', choices=['global', 'avg_1d', 'avg_2d', 'avg_1d_shuffle'])
    
    # aux decoder
    add_arg("--aux-decoder-model", type=str, default='vit_tiny_patch14_dinov2_movq', help='aux decoder model name')
    add_arg("--aux-loss-mask", type=str2bool, default='False', help='compute loss only at mask region')
    add_arg("--aux-hog-decoder", type=str2bool, default=True, help='aux decoder hog decoder')
    add_arg("--aux-dino-decoder", type=str2bool, default=True, help='aux decoder dino decoder')
    add_arg("--aux-clip-decoder", type=str2bool, default=True, help='aux decoder hog decoder')
    add_arg("--aux-supcls-decoder", type=str2bool, default=True, help='aux decoder hog decoder')
    
    # mask modeling
    # make sure drop is 0.0 for not using mask modeling
    add_arg("--enc-token-drop", type=float, default=0.0, help='encoder token drop')
    add_arg("--enc-token-drop-max", type=float, default=0.75, help='maximum drop rate')
    return parser

def main(args):
    """ Train new model """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all subfolders)
        
        exp_idx = int(args.exp_index) if args.exp_index is not None else len(glob(f"{args.results_dir}/*"))

        if args.config is not None:
            model_str_name = '.'.join(args.config.split('/')[-1].split('.')[:-1])
            if model_str_name.startswith('exp'):
                model_str_name = '-'.join(model_str_name.split('-')[1:])
        else:
            model_str_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/exp{exp_idx:03d}-{model_str_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # experiment_config 저장 (순서와 스타일을 그대로 유지하기 위해 round_trip_dump 메서드를 사용하던가)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.YAML().dump(vars(args), f)
        
        wandb_logger = wandb.init(project='tokenizer', name=f'exp{exp_idx:03d}-{model_str_name}')
    else:
        logger = create_logger(None)
        wandb_logger = None

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = VQ_models[args.vq_model](
        image_size=args.image_size,                             # 256
        codebook_size=args.codebook_size,                       #
        codebook_embed_dim=args.codebook_embed_dim,             #
        codebook_l2_norm=args.codebook_l2_norm,                 #
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        vq_loss_ratio=args.vq_loss_ratio,
        kl_loss_weight=args.kl_loss_weight,
        dropout_p=args.dropout_p,
        enc_type=args.enc_type,
        encoder_model=args.encoder_model,
        dec_type=args.dec_type,
        decoder_model=args.decoder_model,
        num_latent_tokens=args.num_latent_tokens,
        enc_tuning_method=args.encoder_tuning_method,
        dec_tuning_method=args.decoder_tuning_method,
        enc_pretrained=args.encoder_pretrained,
        dec_pretrained=args.decoder_pretrained,
        enc_patch_size=args.encoder_patch_size,
        dec_patch_size=args.decoder_patch_size,
        tau=args.tau,
        repa=args.repa,
        repa_model=args.repa_model,
        repa_patch_size=args.repa_patch_size,
        repa_proj_dim=args.repa_proj_dim,
        repa_loss_weight=args.repa_loss_weight,
        repa_align=args.repa_align,
        num_codebooks=args.num_codebooks,
        # encoder mask modeling
        enc_token_drop=args.enc_token_drop,
        enc_token_drop_max=args.enc_token_drop_max,
        # aux decoder model
        aux_dec_model=args.aux_decoder_model,
        aux_loss_mask=args.aux_loss_mask,
        aux_hog_dec=args.aux_hog_decoder,
        aux_dino_dec=args.aux_dino_decoder,
        aux_clip_dec=args.aux_clip_decoder,
        aux_supcls_dec=args.aux_supcls_decoder,
        # to pixel
        to_pixel=args.to_pixel,
    )
    logger.info(f"VQ Model Parameters: {sum(prms.numel() for prms in vq_model.parameters() if prms.requires_grad):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training  ### 수정할 것. EMA는 CPU에 놓을 수 있게 설정 요소 추가하자.
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(prms.numel() for prms in ema.parameters() if prms.requires_grad):,}")
    vq_model = vq_model.to(device)


    vq_loss = VQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        disc_dim=args.disc_dim,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
        perceptual_loss=args.perceptual_loss,
        perceptual_model=args.perceptual_model,
        perceptual_dino_variants=args.perceptual_dino_variants,
        perceptual_weight=args.perceptual_weight,
        perceptual_intermediate_loss=args.perceptual_intermediate_loss,
        perceptural_logit_loss=args.perceptual_logit_loss,
        perceptual_resize=args.perceptual_resize,
        perceptual_warmup=args.perceptual_warmup,
        lecam_loss_weight=args.lecam_loss_weight,
        disc_cr_loss_weight=args.disc_cr_loss_weight,
        use_diff_aug=args.use_diff_aug,
        disc_adaptive_weight=args.disc_adaptive_weight,
        wandb_logger=wandb_logger
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(prms.numel() for prms in vq_loss.discriminator.parameters() if prms.requires_grad):,}")
    
    # scaling lr
    args.lr = args.lr * args.global_batch_size / 256
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups_weight_decay(vq_model, weight_decay=args.weight_decay), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        optimizer_disc = torch.optim.AdamW(param_groups_weight_decay(vq_loss.discriminator, weight_decay=args.weight_decay), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    num_update_steps_per_epoch = len(loader)
    max_train_steps = args.epochs * num_update_steps_per_epoch

    # create lr scheduler
    if args.lr_scheduler == 'none':
        vqvae_lr_scheduler = None
        disc_lr_scheduler = None
    else:
        vqvae_lr_scheduler, _ = create_scheduler(
            sched=args.lr_scheduler,
            optimizer=optimizer,
            patience_epochs=0,
            step_on_epochs=False,
            updates_per_epoch=num_update_steps_per_epoch,
            num_epochs=args.epochs,
            warmup_epochs=args.lr_warmup_epochs,
            min_lr=args.lr * 0.1,
        )
        disc_lr_scheduler, _ = create_scheduler(
            sched=args.lr_scheduler,
            optimizer=optimizer_disc,
            patience_epochs=0,
            step_on_epochs=False,
            updates_per_epoch=num_update_steps_per_epoch,
            num_epochs=args.epochs,
            warmup_epochs=args.lr_warmup_epochs,
            min_lr=args.lr * 0.1,
        )

    logger.info(f"num_update_steps_per_epoch {num_update_steps_per_epoch:,} max_train_steps ({max_train_steps})")

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        
        vq_model.load_state_dict(load_model_state_dict(checkpoint['model']))
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        vq_loss.discriminator.load_state_dict(load_model_state_dict(checkpoint["discriminator"]))
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model) # requires PyTorch 2.0      
        vq_loss.discriminator = torch.compile(vq_loss.discriminator)  
        vq_loss.perceptual_loss = torch.compile(vq_loss.perceptual_loss)
        logger.info("compiling done.")
    
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
                
            imgs = x.to(device, non_blocking=True)

            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                recons_imgs, codebook_loss, info = vq_model(imgs)
                loss_gen = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps+1, 
                                   last_layer=vq_model.module.decoder.last_layer, 
                                   logger=logger, log_every=args.log_every)
            scaler.scale(loss_gen).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

            # discriminator training            
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps+1,
                                    logger=logger, log_every=args.log_every)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()
            
            # # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}/total_steps:{max_train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()
            
                if rank == 0 and wandb_logger is not None:
                    log_dict = {"lr": optimizer.param_groups[0]["lr"], "train_loss": avg_loss}
                    wandb_logger.log(log_dict,
                        step=train_steps
                    )
                
            if train_steps % args.vis_every == 0:
                image = torch.cat([imgs[:4], recons_imgs[:4]], dim=0)
                image = torch.clamp(image, min=-1, max=1)
                image = make_grid((image + 1) / 2, nrow=4, padding=0, pad_value=1.0)
                image = image.permute(1, 2, 0).mul_(255).cpu().numpy()
                image = Image.fromarray(image.astype(np.uint8))

                if rank == 0:
                    wandb_logger.log({"recon_images": [wandb.Image(image)]}, step=train_steps)

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:

                if rank == 0:
                    if args.compile:
                        model_weight = vq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": vq_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    # if not args.no_local_save:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    manage_checkpoints(checkpoint_dir)
                dist.barrier()

            if vqvae_lr_scheduler is not None:
                vqvae_lr_scheduler.step_update(train_steps)
            if disc_lr_scheduler is not None:
                disc_lr_scheduler.step_update(train_steps)


    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    
    parser = build_parser()

    # 1st parse: check config path
    args = parser.parse_args()
    
    # load config and update defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # 2nd parse: final args with config applied
    args = parser.parse_args()
    main(args)
