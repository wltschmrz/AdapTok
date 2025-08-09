# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelling.lpips import LPIPS, LPIPSTimm
from modelling.discriminators import PatchGANDiscriminator, StyleGANDiscriminator, PatchGANMaskBitDiscriminator, DinoDiscriminator
from utils.diff_aug import DiffAugment

import torch.distributed as tdist

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class LeCAM_EMA(object):
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + \
          torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))
    return reg


class VQLoss(nn.Module):
    def __init__(self, 
                 reconstruction_loss='l2', reconstruction_weight=1.0, 
                 mask_weight=1.0,
                 wandb_logger=None):
        super().__init__()

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.mask_weight = mask_weight

        self.wandb_logger = wandb_logger

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        return d_weight.detach()

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step,
                last_layer=None, logger=None, log_every=100, adaptive_weight=False):

        # generator update
        # if optimizer_idx == 0:
        assert optimizer_idx == 0, "check optimizer_idx"

        # reconstruction loss
        rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

        # # discriminator loss
        # if self.use_diff_aug:
        #     reconstructions = DiffAugment(reconstructions.contiguous(), policy='color,translation,cutout_0.2', prob=0.5)
        # logits_fake = self.discriminator(reconstructions.contiguous())
        # generator_adv_loss = self.gen_adv_loss(logits_fake)
             
        # repa loss X
        # mask sparsity loss O
        mask_loss1 = codebook_loss[0]
        mask_loss2 = codebook_loss[1]
        mask_loss3 = codebook_loss[2]

        loss_function = nn.NLLLoss()

        mask_loss1 = -torch.log(1 - codebook_loss[0] + 1e-8)
        mask_loss2 = -torch.log(1 - codebook_loss[1] + 1e-8)
        mask_loss3 = -torch.log(1 - codebook_loss[2] + 1e-8)

        total_mask_loss = codebook_loss[4] if len(codebook_loss) > 4 else mask_loss1 + mask_loss2 + mask_loss3

        # assert adaptive_weight is True
        if adaptive_weight:
            null_loss = self.rec_weight * rec_loss
            mask_ad_w1 = self.calculate_adaptive_weight(null_loss, mask_loss1, last_layer=last_layer)
            mask_ad_w2 = self.calculate_adaptive_weight(null_loss, mask_loss2, last_layer=last_layer)
            mask_ad_w3 = self.calculate_adaptive_weight(null_loss, mask_loss3, last_layer=last_layer)

        else:
            mask_ad_w1 = mask_ad_w2 = mask_ad_w3 = 1
        # disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)


        loss = self.rec_weight * rec_loss # + \
            # mask_ad_w1 * self.mask_weight * mask_loss1 + \
            # mask_ad_w2 * self.mask_weight * mask_loss2 + \
            # mask_ad_w3 * self.mask_weight * mask_loss3
            # perceptual_weight * p_loss + \
            # disc_adaptive_weight * disc_weight * generator_adv_loss
        
        if len(codebook_loss) > 5:
            # other deciders loss
            for item in codebook_loss[5:]:
                loss += item
        
        if global_step % log_every == 0:
            rec_log = self.rec_weight * rec_loss
            logger_string = (
                f"(Generator) rec_loss: {rec_log:.4f}, "
                f"mask_loss1: {mask_loss1:.4f}, mask_loss2: {mask_loss2:.4f}, "
                f"mask_loss3: {mask_loss3:.4f}, total_mask_loss: {total_mask_loss:.4f}, "
                f"mask_adaptive_weight1: {mask_ad_w1:.4f}"
                f"mask_adaptive_weight2: {mask_ad_w2:.4f}"
                f"mask_adaptive_weight3: {mask_ad_w3:.4f}"
            )
            

            if len(codebook_loss) > 5:
                for i in range(5, len(codebook_loss)):
                    logger_string += f", decoder {i:d}: {codebook_loss[i]:.4f}"
            logger.info(logger_string)
            
            if tdist.get_rank() == 0 and self.wandb_logger is not None:
                log_dict = {
                    'rec_loss': rec_log,
                    'mask_loss': total_mask_loss,
                    'mask_loss1': mask_loss1,
                    'mask_loss2': mask_loss2,
                    'mask_loss3': mask_loss3,
                    'total_mask_loss': total_mask_loss,
                    'mask_ad_w1': mask_ad_w1,
                    'mask_ad_w2': mask_ad_w2,
                    'mask_ad_w3': mask_ad_w3,
                }
                if len(codebook_loss) > 5:
                    for i in range(5, len(codebook_loss)):
                        log_dict[f"decoder_{i}"] = codebook_loss[i]
                self.wandb_logger.log(log_dict, step=global_step)
                
        return loss