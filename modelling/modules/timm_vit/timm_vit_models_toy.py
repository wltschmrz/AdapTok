import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
import scipy.stats as stats

import peft
from timm.models import create_model
from timm.layers import trunc_normal_
from modelling.modules.timm_vit.to_pixel import ToPixel
from modelling.modules.timm_vit.vision_transformer import Attention, MoVQNorm, MoVQBlockv2
from modelling.modules.timm_vit.rope_utils import compute_axial_cis, compute_mixed_cis, init_random_2d_freqs, init_t_xy
from modelling.modules.timm_vit.dyvit_toy import VisionTransformerDiffPruning


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


class TimmViTEncoder(nn.Module):
    def __init__(self, in_channels=3, num_latent_tokens=16,
                 model_name='vit_tiny_patch16_224',
                 model_kwargs={'img_size': 128, 'patch_size': 16, 'drop_path_rate': 0.0,},
                 pretrained=False, tuning_method='full', tuning_kwargs={'r': 8},
                 rope_theta=100.0, rope_mixed=True, use_rope=True, use_ape=False,
                 token_drop=0.4,
                 token_drop_max=0.6,
                 base_img_size=128
                 ):
        super().__init__()

        self.model_name = model_name
        assert model_name in ['vit_tiny_patch16_224'], f"{model_name} not found"

        # parameters
        self.num_latent_tokens = num_latent_tokens

        print(f"model_kwargs:")
        print(model_kwargs)
        
        ######
        # load model
        base_rate = 0.9
        PRUNING_LOC = [2, 4, 6]
        model = VisionTransformerDiffPruning(
            img_size=128,
            patch_size=16, embed_dim=48, depth=8, num_heads=4, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC
            )
        model.num_prefix_tokens=0

        self.img_size = model_kwargs['img_size']
        self.patch_size = model_kwargs['patch_size']
        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        
        # tuning method
        assert tuning_method == 'full'
        self.model = model

        assert self.num_latent_tokens
        # latent tokens
        self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        nn.init.normal_(self.latent_tokens, std=.02)
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        trunc_normal_(self.latent_pos_embed, std=.02)

        # token drop
        self.token_drop = token_drop > 0.0
        if self.token_drop:
            # self.mask_ratio_generator = stats.truncnorm((1.0 - token_drop) / 0.25, 1.0 / 0.25, loc=1.0, scale=0.25)
            self.mask_ratio_generator = stats.truncnorm((token_drop - token_drop_max) / 0.25, 0, loc=token_drop_max, scale=0.25)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
            nn.init.normal_(self.mask_token, std=.02)

        # rope
        self.use_ape = use_ape
        self.use_rope = use_rope
        if self.use_rope:
            self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        
        assert self.rope_mixed and self.use_rope, f"{self.rope_mixed}, {self.use_rope}"
        self.compute_cis = partial(compute_mixed_cis, num_heads=model.num_heads)
        freqs = []
        for i, _ in enumerate(model.blocks):
            freqs.append(init_random_2d_freqs(dim=model.embed_dim // model.num_heads, num_heads=model.num_heads, theta=self.rope_theta))
        freqs = torch.stack(freqs, dim=1).view(2, len(model.blocks), -1)
        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
        
        if base_img_size != model_kwargs['img_size']:
            t_x, t_y = init_t_xy(end_x = base_img_size // model_kwargs['patch_size'] , end_y =  base_img_size //  model_kwargs['patch_size'] )
        else:
            t_x, t_y = init_t_xy(end_x = model_kwargs['img_size'] // model_kwargs['patch_size'] , end_y =  model_kwargs['img_size'] //  model_kwargs['patch_size'] )
        self.register_buffer('freqs_t_x', t_x)
        self.register_buffer('freqs_t_y', t_y)

        if not self.use_ape:
            for b in self.model.blocks:
                b.attn.flash_attn = False
        
    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'latent_tokens', 'latent_pos_embed', 'freqs']

    def sample_orders(self, bsz, seq_len):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward(self, x, return_mask=False, gumbel_tau=1.0, ):

        # get tokens
        B, _, H, W = x.shape
        x = self.model.patch_embed(x)

        if self.token_drop and self.training:
            orders = self.sample_orders(bsz=x.size(0), seq_len=x.size(1)).to(x.device)
            mask = self.random_masking(x, orders).unsqueeze(-1)
            x = torch.where(mask.bool(), self.mask_token, x)
        else:
            mask = None
        
        assert not 'eva02' in self.model_name
        # x = self.model._pos_embed(x)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        patch_length = x.size(1)

        # insert latent tokens
        z = self.latent_tokens.expand(B, -1, -1)
        x = torch.cat([x, z + self.latent_pos_embed], dim=1)
            
        # pre layer norm
        x = self.model.norm(x)
        
        p_count = 0
        losses = []
        prev_decision = torch.ones(B, patch_length + self.num_latent_tokens, 1, dtype=x.dtype, device=x.device)
        patch_policy = torch.ones(B, patch_length, 1, dtype=x.dtype, device=x.device)
        # if self.use_ape: 
        #     for i, blk in enumerate(self.model.blocks):
        #         x = blk(x)
        assert self.rope_mixed and self.use_rope
        if self.freqs_t_x.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
            t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
            t_x, t_y = t_x.to(x.device), t_y.to(x.device)
        else:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
        freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        
        for i , blk in enumerate(self.model.blocks):
            if i in self.model.pruning_loc:
                # latent_x = x[:, -self.num_latent_tokens:]
                logits = self.model.score_predictor[p_count](
                    x, 
                    patch_policy=prev_decision[:, :-self.num_latent_tokens, :],
                    latent_policy=prev_decision[:, -self.num_latent_tokens:, :],
                    )  # (BL) <- (BN2)
                # print(f"##### {logits.shape}")  # ##### torch.Size([128, 16])
                if self.training:
                    p_soft = F.gumbel_softmax(logits, tau=gumbel_tau, hard=False, dim=1)
                    # print(f"##### {p_soft.shape}")  # torch.Size([128, 16])
                    large_pos = torch.argmax(p_soft, dim=1, keepdim=True)  # (B, 1)

                    # print(f"##### {large_pos.shape}")  ##### torch.Size([128])

                    cumsum_p = torch.cumsum(p_soft, dim=1)
                    keep_soft = 1.0 - cumsum_p
                    # print(f"##### {keep_soft.shape}")  ##### torch.Size([128, 16])

                    # mask_loss = torch.sum(keep_soft.sum(dim=1, keepdim=True) / self.num_latent_tokens) / B
                    mask_loss = keep_soft.sum(dim=1, keepdim=True) / self.num_latent_tokens
                    # print(f"##### {mask_loss.shape}")    ##### torch.Size([128, 1])
                    pos = torch.gather(keep_soft, 1, large_pos)
                    # print(f"##### {pos.shape}")  ##### torch.Size([128, 128])
                    keep_hard = (keep_soft >= pos).float()
                    # print(f"##### {keep_hard.shape}")
                    keep_mask = (keep_hard - keep_soft).detach() + keep_soft   # (B, L)
                    # print(f"##### {keep_mask.shape}")
                    hard_keep_decision = keep_mask.reshape(B, self.num_latent_tokens, 1) * prev_decision[:, -self.num_latent_tokens:, :]
                    # mask_loss  (B, 1)

                    # losses.append(mask_loss.reshape(B, 1))
                    losses.append(mask_loss.mean())
                    policy = torch.cat([patch_policy, hard_keep_decision], dim=1)
                    x = blk(x, policy=policy, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                    prev_decision = policy
                else:
                    with torch.no_grad():
                        p_soft = F.softmax(logits, dim=1)
                        large_pos = torch.argmax(p_soft, dim=1)

                        cumsum_p = torch.cumsum(p_soft, dim=1)
                        keep_soft = 1.0 - cumsum_p

                        pos = keep_soft[:, large_pos].item()
                        keep_hard = (keep_soft >= pos).float()
                        keep_mask = (keep_hard - keep_soft).detach() + keep_soft   # (B, L)
                        hard_keep_decision = keep_mask.reshape(B, self.num_latent_tokens, 1) * prev_decision[:, -self.num_latent_tokens:, :]
                        
                        now_policy = torch.cat([patch_policy, hard_keep_decision], dim=1)
                        x = batch_index_select(x, now_policy)
                        prev_decision = batch_index_select(prev_decision, now_policy)
                        x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                p_count += 1

        # x = self.model.blocks(x)
        x = self.model.norm(x)

        # get z tokens as out
        out = x[:, -self.num_latent_tokens:]
        
        if return_mask:
            return out, mask, losses
        else:
            return out

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

class TimmViTDecoder(nn.Module):
    def __init__(self, in_channels=3,
                 model_name='vit_tiny_patch16_224',
                 model_kwargs={'img_size': 128, 'patch_size': 16, 'drop_path_rate': 0.0, 'embed_dim': 48}, pretrained=True,
                 tuning_method='lora', tuning_kwargs={'r': 8},
                 num_latent_tokens=16, to_pixel='linear',
                 codebook_embed_dim=32,
                 rope_theta=100.0, rope_mixed=False, use_rope=False, use_ape=True,
                 cls_token=True,
                 base_img_size=128,
                 ):
        super().__init__()

        # model_kwargs['num_latent_tokens'] = num_latent_tokens
        # model_kwargs['class_token'] = cls_token
        model = create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )
        
        self.patch_size = model_kwargs['patch_size']
        self.embed_dim = model.embed_dim = 48
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens
        
        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        # latent tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        trunc_normal_(self.latent_pos_embed, std=.02)

        # to pixel
        self.to_pixel = ToPixel(to_pixel=to_pixel, img_size=model_kwargs['img_size'], in_channels=in_channels,
                                in_dim=model.embed_dim, patch_size=model_kwargs['patch_size'])

        
        self.use_ape = use_ape
        self.use_rope = use_rope
        if self.use_rope:
            self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        
        if self.rope_mixed and self.use_rope:
            self.compute_cis = partial(compute_mixed_cis, num_heads=model.num_heads)
            
            freqs = []
            for i, _ in enumerate(model.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=model.embed_dim // model.num_heads, num_heads=model.num_heads, theta=self.rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(model.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            if base_img_size != model_kwargs['img_size']:
                t_x, t_y = init_t_xy(end_x = base_img_size // model_kwargs['patch_size'] , end_y =  base_img_size //  model_kwargs['patch_size'] )
            else:
                t_x, t_y = init_t_xy(end_x = model_kwargs['img_size'] // model_kwargs['patch_size'] , end_y =  model_kwargs['img_size'] //  model_kwargs['patch_size'] )
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        elif not self.rope_mixed and self.use_rope:
            self.compute_cis = partial(compute_axial_cis, dim=model.embed_dim//model.num_heads, theta=rope_theta)
            
            freqs_cis = self.compute_cis(end_x = model_kwargs['img_size'] // model_kwargs['patch_size'] , end_y = model_kwargs['img_size'] //  model_kwargs['patch_size'] )
            self.freqs_cis = freqs_cis
            
        if not self.use_ape:
            for b in self.model.blocks:
                b.attn.flash_attn = False


        if 'movq' in model_name:
            self.use_movq = True 
            self.model.norm = MoVQNorm(codebook_embed_dim, model.embed_dim)

            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.model.blocks:
                if isinstance(block, MoVQBlockv2):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            if isinstance(self.model.norm, MoVQNorm):
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].bias, 0)
        else:
            self.use_movq = False 
            

        self.cls_token = cls_token
        if not cls_token:
            self.model.cls_token = None
            self.num_prefix_tokens -= 1
            self.model.num_prefix_tokens -= 1
            
    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'mask_token', 'latent_pos_embed', 'freqs']

    @property
    def last_layer(self):
        return self.to_pixel.get_last_layer()


    def forward(self, z, interpolate_zq=None, H=None, W=None):

        if H is None:
            num_img_tokens = self.num_img_tokens
            H = W = int(math.sqrt(num_img_tokens)) * self.patch_size
        else:
            num_img_tokens = H * W // self.patch_size ** 2

        # mask tokens
        if self.num_latent_tokens:
            if H is None:
                x = self.mask_token.expand(z.size(0), num_img_tokens, -1)
            else:
                x = self.mask_token.expand(z.size(0), H * W // self.patch_size ** 2, -1)
        else:
            x = z 
            
        x = self.model._pos_embed(x, use_ape=self.use_ape)
        x = self.model.patch_drop(x)
        
        z = z + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)

        x = self.model.norm_pre(x)
        
        
        if self.use_ape: 
            for i, blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq=interpolate_zq, num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x)
                
        elif self.rope_mixed and self.use_rope:
            if self.freqs_t_x.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)

        else:
            if self.freqs_cis.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq,  freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)      

        if self.use_movq:
            x = self.model.norm(x, interpolate_zq,  num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
        else:
            x = self.model.norm(x)

        x = x[:, self.num_prefix_tokens:self.num_img_tokens + self.num_prefix_tokens]

        out = self.to_pixel(x)

        return out


if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from dyvit import VisionTransformerDiffPruning

    encoder = TimmViTEncoder(num_latent_tokens=256)

    TimmViTEncoder(
        in_channels=3, num_latent_tokens=256,
        model_name='vit_tiny_patch16_224',  # 'vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
        model_kwargs={'img_size': 256, 'patch_size': 16, 'drop_path_rate': 0.0},  # enc_drop_path_rate},
        pretrained=False,
        tuning_method='full',
        tuning_kwargs={'r': 8},
        use_ape=True, use_rope=False, rope_mixed=False, rope_theta=10.0,
        token_drop=0.0,
        token_drop_max=0.6,
        base_img_size=256
        )



    base_rate = 0.9
    SPARSE_RATIO = [base_rate, base_rate - 0.2, base_rate - 0.4]
    PRUNING_LOC = [3, 6, 9]
    KEEP_RATE = [SPARSE_RATIO[0], SPARSE_RATIO[0] ** 2, SPARSE_RATIO[0] ** 3]
    encoder2 = model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True
        )

    print(encoder)
    print('='*100)
    print(encoder2)


    # encoder = TimmViTEncoder(num_latent_tokens=256)
    # decoder = TimmViTDecoder(num_latent_tokens=256)
    
    # x = torch.randn(1, 3, 224, 224)
    
    # o = encoder(x)
    # print(o.shape)
    # r = decoder(o)





