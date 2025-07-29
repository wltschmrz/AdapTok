# [ICML25 Spotlight] Masked Autoencoders Are Effective Tokenizers for Diffusion Models


## Change Logs
* [05/28/2025] Training code of MAETok released. 
* [02/05/2025] 512 and 256 SiT models and MAETok released. LightingDiT models will be updated and we will update the training scripts soon.
* [12/19/2024] 512 SiT models and DiT models released. We also updated the training scripts.

## Models

### MAETok Tokenizer (rFID 기반 성능)


| Tokenizer 	| Image Size | rFID 	| Huggingface 	|
|:---:	| :---:	| :---:	|:---:	|
| MAETok-B-128 	| 256 | 0.48 	| [Model Weight](https://huggingface.co/MAETok/maetok-b-128) 	|
| MAETok-B-128-512 	| 512 | 0.62 	| [Model Weight](https://huggingface.co/MAETok/maetok-b-512) 	|


### SiT-XL Models on MAETok

| Genenerative Model | Image Size	| Tokenizer 	| gFID (w/o CFG) |	gFID (w/ CFG)| Huggingface 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| SiT-XL 	| 256 | MAETok-B-128 	| 2.31 	| 1.67 | [Model Weight](https://huggingface.co/MAETok/sit-xl_maetok-b-128) 	|
| SiT-XL 	| 512 | MAETok-B-128-512	| 2.79 	| 1.69 | [Model Weight](https://huggingface.co/MAETok/sit-xl_maetok-b-128-512) 	|


## Setup
```
conda create -n maetok python=3.10 -y; conda activate maetok
# torch 먼저 설치할 것.
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
cd MAETok_for_128/ && pip install -r requirements.txt; pip install opencv-python-headless
# 뒤에도 해주자 hog에서 필요하대.
wget https://www.image-net.org/data/tiny-imagenet-200.zip && unzip tiny-imagenet-200

# imagenet10p
gdown 1XySOXm-eh8zcLAi1vM8p7axx9HaXh148
tar -xzf ./imagenet10p_train.tar.gz

https://drive.google.com/file/d/19-cXr1ug0uloGWqjKG9zvr5bxmMtiQ4M/view?usp=drive_link  # grid
https://drive.google.com/file/d/1XySOXm-eh8zcLAi1vM8p7axx9HaXh148/view?usp=drive_link  # imagenet10p
```

## Training 

**Train Tokenizer**
```
torchrun --nproc_per_node=8 train/train_tokenizer.py --config configs/maetok-b-64.yaml
torchrun --nproc_per_node=1 train/train_tokenizer.py --config configs/maetok-b-64.yaml --global-batch-size 32 --data-path tiny-imagenet-200/train --image-size 256
torchrun --nproc_per_node=1 train/train_tokenizer.py --config configs/maetok-b-64.yaml --global-batch-size 1 --data-path tiny-imagenet-200/train --image-size 256 --aux-hog-decoder False --aux-dino-decoder False --aux-clip-decoder False --aux-supcls-decoder False

nohup torchrun --nproc_per_node=4 train/train_tokenizer.py \
  --config configs/maetok-b-64.yaml \
  --global-batch-size 128 \
  --data-path data/imagenet10p/train \
  --image-size 256 > train.log 2>&1 &

torchrun --nproc_per_node=1 train/train_tokenizer_toy.py

```

**Train SiT**
```
torchrun --nproc_per_node=8 train/train_sit.py \
--report-to="wandb" \
--allow-tf32 \ 
--mixed-precision="bf16" \
--seed=0 \
--path-type="linear" \
--prediction="v" \
--weighting="lognormal" \
--model="SiT-XL/1" \
--vae-model='softvq-l-64' \
--output-dir="experiments/sit" \
--exp-index=1 \
--data-dir=./imagenet/train
```

**Train DiT**
```
torchrun --nproc_per_node=8 train/train_dit.py \
--data-path ./imagenet/train \
--results-dir experiments/dit \
--model DiT-XL/1 \
--epochs 1400 \
--global-batch-size 256 \
--mixed-precision bf16 \
--vae-model='softvq-l-64'  \
--noise-schedule cosine  \
--disable-compile
```

## Inference


**Reconstruction**
```
torchrun --nproc_per_node=8 inference/reconstruct_vq.py \
--data-path ./ImageNet/val \
--vq-model MAETok/maetok-b-128
```


**SiT Generation**
```
torchrun --nproc_per_node=8 inference/generate_sit.py --tf32 True --model SoftVQVAE/sit-xl_softvq-b-64 --cfg-scale 1.75 --path-type cosine --num-steps 250 --guidance-high 0.7 --vae-model softvq-l-64
```

**DiT Generation**
```
torchrun --nproc_per_node=8 inference/generate_dit.py --model SoftVQVAE/dit-xl_softvq-b-64--cfg-scale 1.75 --noise-schedule cosine --num-sampling-steps 250 --vae-model softvq-l-64
```


**Evaluation**
We use [ADM](https://github.com/openai/guided-diffusion/tree/main) evaluation toolkit to compute the FID/IS of generated samples


**GMM Fitting**
```
# save training latent first
torchrun --nproc_per_node=8 inference/cache_latent.py --dataset imagenet --data-path imagenet/train --sample-dir saved_latent --vae-name maetok-b-128
# run gmm
python inference/gmm_fit.py --use_gpu 0 --exp  maetok-b-128 --n_iter 500 --samples_per_class 100 --components 5 10 50 100 200 300
```

## Reference
```
@inproceedings{chen2025maetok,
    title={Masked Autoencoders Are Effective Tokenizers for Diffusion Models},
    author={Hao Chen and Yujin Han and Fangyi Chen and Xiang Li and Yidong Wang and Jindong Wang and Ze Wang and Zicheng Liu and Difan Zou and Bhiksha Raj},
    booktitle={ICML},
    year={2025},
}

```
