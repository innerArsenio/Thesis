import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import schedulefree
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from models.sit_not_yet_messed_up import SiT_models as SiT_models_not_yet_messed_up
from models.sit_improved_maybe import SiT_models as SiT_models_improved_maybe
from models.sit_with_additional_tokens import SiT_models as SiT_models_with_additional_tokens
from models.sit_with_additional_tokens import SiT_Patch_and_Unpatchifier as Patch_and_Unpatchifier
from models.sit_wild import SiT_models as SiT_models_wild
from loss_with_attn import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

from sklearn.metrics import balanced_accuracy_score
import torch.nn as nn
from torch import optim
import timm
from optparse import OptionParser
from torchvision import transforms
from Explicd.dataset.isic_dataset import SkinDataset
from Explicd.model import (ExpLICD_ViT_L, ExpLICD, ExpLICD_ViT_L_Multiple_Prompts, ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens, PatchSelectorCNNConscise, 
                           ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Plus_SuperPixels, ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Plus_Representation_Learning, 
                           ExpLICD_ViT_L_Classic, ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Branching, ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Cascade,
                           ExpLICD_ViT_L_Classic_with_Spatial_Bias)
from Explicd.concept_dataset import (explicid_isic_dict, explicid_isic_dict_mine, explicid_idrid_dict, explicid_idrid_edema_dict, explicid_busi_dict, explicid_busi_soft_smooth_dict, 
                                     explicid_isic_minimal_dict, explicid_isic_binary_dict)
import Explicd.utils as utils
from sklearn.metrics import f1_score
import random
import kornia
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image

imgs_save_dir = "Batch_images"
os.makedirs(imgs_save_dir, exist_ok=True)


logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

CONCEPT_LABEL_MAP_ISIC = [
            [3, 0, 0, 3, 3, 0, 2], # AKIEC
            [2, 0, 2, 2, 2, 0, 1], # BCC
            [4, 2, 1, 4, 4, 1, 3], # BKL
            [5, 1, 1, 5, 5, 1, 0], # DF
            [0, 0, 0, 0, 0, 0, 0], # MEL
            [1, 1, 1, 1, 1, 1, 0], # NV
            [6, 3, 1, 6, 1, 2, 0], # VASC
        ]

CONCEPT_LABEL_MAP_ISIC_MINIMAL = [
            [0, 0, 0, 0, 0, 0, 0], # AKIEC
            [1, 1, 1, 1, 1, 1, 1], # BCC
            [2, 2, 2, 2, 2, 2, 2], # BKL
            [3, 3, 3, 3, 3, 3, 3], # DF
            [4, 4, 4, 4, 4, 4, 4], # MEL
            [5, 5, 5, 5, 5, 5, 5], # NV
            [6, 6, 6, 6, 6, 6, 6], # VASC
        ]

CONCEPT_LABEL_MAP_ISIC_MINE = [
    # Actinic Keratoses
    [2, 3, 2, 0, 3, 1],
    # Basal Cell Carcinoma
    [2, 3, 2, 1, 3, 2],
    # Benign Keratosis-like Lesions
    [2, 2, 2, 1, 2, 1],
    # Dermatofibroma
    [1, 1, 1, 0, 1, 0],
    # Melanoma
    [3, 4, 3, 2, 4, 2],
    # Melanocytic Nevus
    [1, 1, 1, 0, 1, 0],
    # Vascular Lesions
    [0, 0, 0, 0, 0, 2],
]

CONCEPT_LABEL_MAP_IDRID = [
    # No DR
    [0, 0, 0, 0, 0], 
    # Mild DR
    [1, 1, 1, 0, 0],
    # Moderate DR
    [2, 2, 2, 0, 1], 
    # Severe DR
    [3, 3, 3, 0, 2],
    # Proliferative DR
    [3, 3, 3, 1, 3], 
]

CONCEPT_LABEL_MAP_IDRID_EDEMA = [
    # No Edema
    [0, 0, 0, 0, 0, 0], 
    # Mild Edema
    [1, 1, 1, 1, 0, 1],
    # Severe Edema
    [2, 1, 1, 2, 1, 2], 
]


CONCEPT_LABEL_MAP_BUSI = [
    # Benign
    [1, 1, 1, 1, 1, 0],
    # Malignant
    [2, 2, 2, 2, 2, 1],  
    # Normal
    [0, 0, 0, 0, 0, 0],  
]

CONCEPT_LABEL_MAP_BUSI_SOFT_SMOOTH = [
    # Benign
    [[0.2, 0.6, 0.2], [0.2, 0.6, 0.1, 0.1], [0.0, 0.8, 0.1, 0.1], [0.6, 0.1, 0.3], [0.1, 0.9] , [0.8, 0.2]],
    # Malignant
    [[0.0, 0.1, 0.9], [0.0, 0.1, 0.2, 0.7], [0.0, 1.0, 0.0, 0.0], [0.1, 0.8, 0.1], [0.8, 0.2], [0.2, 0.8]],  
    # Normal
    [[0.8, 0.2, 0.0], [1.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.7, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],  
]

CONCEPT_LABEL_MAP_ISIC_SOFT_SMOOTH = [
    # Actinic Keratoses
    [[0.7 , 0.3], [0.6, 0.4], [0.5, 0.5], [0.1, 0.9], [0.7, 0.3] , [0.7, 0.3]],
    # Basal Cell Carcinoma
    [[0.5, 0.5], [0.4, 0.6], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7] , [0.3, 0.7]],  
    # Benign Keratosis-like Lesions
    [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.2, 0.8], [0.8, 0.2] , [0.8, 0.2]], 
    # Dermatofibroma
    [[0.8, 0.2], [0.7, 0.3], [0.7, 0.3], [0.4, 0.6], [0.6, 0.4] , [0.6, 0.4]],
    # Melanoma
    [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.3, 0.7], [0.5, 0.5] , [0.4, 0.6]],  
    # Melanocytic Nevus
    [[0.7, 0.3], [0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.8, 0.2] , [0.2, 0.8]],
    # Vascular Lesions
    [[0.8, 0.2], [0.7, 0.3], [0.7, 0.3], [0.6, 0.4], [0.1, 0.9] , [0.6, 0.4]]
]


CONCEPT_LABEL_MAP_DICT = {
    'ISIC': CONCEPT_LABEL_MAP_ISIC,
    'ISIC_MINE': CONCEPT_LABEL_MAP_ISIC_MINE,
    'ISIC_MINIMAL':CONCEPT_LABEL_MAP_ISIC_MINIMAL,
    'ISIC_SOFT':CONCEPT_LABEL_MAP_ISIC_SOFT_SMOOTH,

    'IDRID': CONCEPT_LABEL_MAP_IDRID,
    'IDRID_EDEMA': CONCEPT_LABEL_MAP_IDRID_EDEMA,

    'BUSI': CONCEPT_LABEL_MAP_BUSI,
    'BUSI_SOFT': CONCEPT_LABEL_MAP_BUSI_SOFT_SMOOTH
    
}

NUM_OF_CRITERIA = {
    'ISIC': 7,
    'ISIC_MINE': 6,
    'ISIC_MINIMAL': 7,
    'ISIC_SOFT': 6,

    'IDRID': 5,
    'IDRID_EDEMA': 6,

    'BUSI': 6,
    'BUSI_SOFT': 6
}

LIST_OF_TASKS = ['ISIC', 'ISIC_MINE', 'ISIC_MINIMAL', 'ISIC_SOFT', 'IDRID', 'IDRID_EDEMA', 'BUSI', 'BUSI_SOFT']

#TASK='IDRID_EDEMA'

NUM_OF_CLASSES= {
    'ISIC': 7,
    'ISIC_MINE':7,
    'ISIC_MINIMAL':7,
    'ISIC_SOFT':7,

    'IDRID': 5,
    'IDRID_EDEMA':3,

    'BUSI': 3,
    'BUSI_SOFT':3
}

CONCEPTS= {
    'ISIC': explicid_isic_dict,
    'ISIC_MINE': explicid_isic_dict_mine,
    'ISIC_MINIMAL':explicid_isic_minimal_dict,
    'ISIC_SOFT':explicid_isic_binary_dict,

    'IDRID': explicid_idrid_dict,
    'IDRID_EDEMA': explicid_idrid_edema_dict,

    'BUSI': explicid_busi_dict,
    'BUSI_SOFT': explicid_busi_soft_smooth_dict
}
CONCEPT_HARDNESS_LIST_OPTIONS=["hard","soft_equal","soft_smarter"]

#DO_MUDDLE_CHECK=True
#ADD_GAUSSIAN_NOISE=False
DO_LOGITS_SIMILARITY=False
#CONCEPT_HARDNESS="soft_equal"
DO_CONTR_LOSS = False
noise_levels = [0, 5, 10, 15, 20]

SAVING_BASED_ON_STEP=False
SAVING_BASED_ON_SCORE=False

do_val_check=True

def create_muddled_dataloader(dataset, batch_size, num_workers):
    """Creates a DataLoader with standard parameters"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

#CONCEPT_LABEL_MAP = CONCEPT_LABEL_MAP_DICT[TASK]

DEBUG = False

def set_seed_mine(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


rotation_transform = transforms.RandomRotation(degrees=15)
translation_transform = transforms.RandomAffine(degrees=3, translate=(0.05, 0.05))
color_jitter_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
blur_transform = transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.2, 5))


img_transform_function_dict = {
    "raw": None,
    "blurred": blur_transform,
    "rotated": rotation_transform,
    "translated":translation_transform,
    "color_jittered":color_jitter_transform
}

#TRANSFORM_FUNCTIONS=["raw", "blurred", "rotated", "translated", "color_jittered"]
TRANSFORM_FUNCTIONS=["raw"]
def validation(explicd, model, dataloader, exp_val_transforms, explicd_only=0, accelerator=None):
    net = explicd
    if explicd_only==0:
        sit_model = model

    net.eval()
    net.zero_grad(set_to_none=True)
    if explicd_only==0:
        sit_model.eval()
        sit_model.zero_grad(set_to_none=True)

    exp_pred_list = np.zeros((0), dtype=np.uint8)
    exp_cnn_critical_pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    agg_visual_tokens_list=[]

    with torch.no_grad():
        for _, (raw_image, _, y) in enumerate(dataloader):
            raw_image = raw_image.to("cuda")
            y = y.to("cuda")
            labels = y

            imgs_for_explicid=prepare__imgs_for_explicid(raw_image, exp_val_transforms).to("cuda")
            explicd_return_dict = explicd(imgs_for_explicid)
            # cls_with_te_logits = explicd_return_dict["cls_with_te_logits"]
            # cls_logits = explicd_return_dict["cls_logits"]
            cls_logits = explicd_return_dict["cls_logits"]
            agg_critical_visual_tokens = explicd_return_dict["agg_critical_visual_tokens"]
            agg_trivial_visual_tokens = explicd_return_dict["agg_trivial_visual_tokens"]
            #cnn_logits_critical = explicd_return_dict["cnn_logits_critical"]
            attention_weights = explicd_return_dict["attn_critical_weights"]
            cnn_logits_critical = explicd_return_dict["cnn_logits_critical"]
            longer_visual_tokens = torch.cat([agg_critical_visual_tokens, agg_trivial_visual_tokens], dim=1)
            agg_visual_tokens_list.append(longer_visual_tokens)

            _, exp_label_pred = torch.max(cls_logits, dim=1)
            _, exp_cnn_critical_label = torch.max(cnn_logits_critical, dim=1)
            exp_pred_list = np.concatenate((exp_pred_list, exp_label_pred.cpu().numpy().astype(np.uint8)), axis=0)
            exp_cnn_critical_pred_list = np.concatenate((exp_cnn_critical_pred_list, exp_cnn_critical_label.cpu().numpy().astype(np.uint8)), axis=0)
            gt_list = np.concatenate((gt_list, labels.cpu().numpy().astype(np.uint8)), axis=0)
        

            

        # imgs_for_explicid = accelerator.gather(imgs_for_explicid.to(torch.float32))
        # #print(f"imgs_for_explicid shape {imgs_for_explicid.shape}")
        # accelerator.log({f"imgs_for_explicid": wandb.Image(array2grid(imgs_for_explicid))})

        # ###################################
        B, C, H, W = imgs_for_explicid.shape
        num_tokens, num_patches = attention_weights.shape[1], attention_weights.shape[2]
        for t in range(num_tokens//2):
            print(f"attention_weights token {t} {attention_weights[0, t, :]}")
            print(f"attention_weights token {t} {attention_weights[0, t, :].sum()}")
            print(f"attention_weights token {t} {attention_weights[1, t, :]}")
        # Assuming the image has been split into non-overlapping patches of size patch_size
        grid_size = int(np.sqrt(num_patches))  # Assuming square grid of patches
        
        # Reshape the attention weights to (B, num_tokens, grid_size, grid_size)
        attention_weights = attention_weights.view(B, num_tokens, grid_size, grid_size)

        # Scale attention weights to the image grid (224x224)
        attention_weights_resized = F.interpolate(attention_weights, size=(H, W), mode='bilinear', align_corners=False)
        
        # The attention weights are now of shape (B, num_tokens, H, W)
        # Normalize the attention weights to [0, 1]
        attention_weights_resized = torch.clamp(attention_weights_resized, 0, 1)
        
        # Now apply the attention weights to the images
        # Repeat the attention map for each channel (RGB)
        #imgs_with_attention = imgs_for_explicid.clone()  # Make a copy to modify
        imgs_with_attention_token_list=[]
        
        # Multiply each channel of the image with the attention weights (apply it per pixel)
        for t in range(num_tokens):
            # Select attention map for current token, it's of shape (B, H, W)
            att_map = attention_weights_resized[:, t, :, :]  # (B, H, W)
            
            # Repeat the attention map for each channel (to apply it to RGB channels)
            att_map_expanded = att_map.unsqueeze(1)  # Shape: (B, 1, H, W)
            #print(att_map_expanded[0,0,:,:])
            
            # Apply the attention map to the image channels
            imgs_with_attention_token =  att_map_expanded * 255

            imgs_with_attention_token = accelerator.gather(imgs_with_attention_token.to(torch.float32))
            imgs_with_attention_token_list.append(imgs_with_attention_token)
            #print(f"imgs_for_explicid shape {imgs_for_explicid.shape}")
                
        accelerator.log({
                        f"imgs_for_explicid": wandb.Image(array2grid(imgs_for_explicid)),
                        f"imgs_for_explicid token {0} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[0])),
                        f"imgs_for_explicid token {1} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[1])),
                        f"imgs_for_explicid token {2} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[2])),
                        f"imgs_for_explicid token {3} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[3])),
                        f"imgs_for_explicid token {4} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[4])),
                        #f"imgs_for_explicid token {5} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[5])),
                        #f"imgs_for_explicid token {6} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[6]))
                            })
        
        ###################################
        exp_BMAC = balanced_accuracy_score(gt_list, exp_pred_list)
        exp_correct = np.sum(gt_list == exp_pred_list)
        exp_acc = 100 * exp_correct / len(exp_pred_list)
        exp_val_f1 = f1_score(gt_list, exp_pred_list, average='macro')
        exp_cnn_critical_val_f1 = f1_score(gt_list, exp_cnn_critical_pred_list, average='macro')

        expl_scores={
            "BMAC":exp_BMAC,
            "f1":exp_val_f1,
            "Acc":exp_acc,
            "cnn critical f1": exp_cnn_critical_val_f1
        }

        tokens_and_gt={
            #"tokens":torch.stack(agg_visual_tokens_list, dim=0),
            "tokens":torch.cat(agg_visual_tokens_list, dim=0),
            "gt":gt_list
        }

    return expl_scores, tokens_and_gt


class DualRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        # imgs is a tuple (image1, image2)
        img1, img2 = imgs

        # Apply the random vertical flip with the same probability to both images
        if torch.rand(1) < self.p:
            img1 = transforms.functional.vflip(img1)
            #img2 = transforms.functional.vflip(img2)
        
        return img1, img2
    
class DualRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        # imgs is a tuple (image1, image2)
        img1, img2 = imgs

        # Apply the random vertical flip with the same probability to both images
        if torch.rand(1) < self.p:
            img1 = transforms.functional.hflip(img1)
            #img2 = transforms.functional.hflip(img2)
        
        return img1, img2
    
class DualGaussianNoise:
    def __init__(self, p=0.5, mean=0, std=0):
        self.p = p
        self.mean=mean
        self.std=std

    def __call__(self, imgs):
        # imgs is a tuple (image1, image2)
        img1, img2 = imgs

        # Apply the random vertical flip with the same probability to both images
        if torch.rand(1) < self.p:
            img1 = img1.float()
            img1 = img1 + torch.normal(self.mean, self.std, size=img1.shape, device=img1.device)
            img1 = torch.clamp(img1, 0, 255).to(torch.uint8)
        
        return img1, img2


def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        #print("clip frozen encoder is used")
        x = x / 255.
        ##x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = torch.nn.functional.interpolate(x, 448, mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')

    return x

def prepare__imgs_for_explicid(x, explicid_img_transforms):
    images_transformed = torch.stack([explicid_img_transforms(img) for img in x])
    return images_transformed

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def patches_to_images(patches, patch_size=14):
    """
    Splits an image (3, 224, 224) into patches of size (256, D), where D = 14*14*3.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (3, 224, 224).
        patch_size (int): Size of each patch along one dimension.

    Returns:
        torch.Tensor: Tensor of shape (256, D) where D = patch_size * patch_size * 3.
    """
    # Ensure image is a tensor with shape (B, 3, 224, 224)
    B, num_patches, D = patches.shape
    image_size = int(num_patches ** 0.5)*patch_size
    grid_size = image_size//patch_size

    patches = patches.view(B, num_patches, 3, patch_size, patch_size)
    patches = patches.view(B, grid_size, grid_size, 3, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    images = patches.contiguous().view(B, 3, image_size, image_size)
    return images

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # Set the seed
    set_seed_mine(42)
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    # ['ISIC', 'ISIC_MINE', 'ISIC_MINIMAL', 'ISIC_SOFT', 'IDRID', 'IDRID_EDEMA', 'BUSI', 'BUSI_SOFT']
    TASK = args.task
    DO_MUDDLE_CHECK = args.muddle_check
    ADD_GAUSSIAN_NOISE = args.add_gaussian

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        #checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        checkpoint_dir_step_based=f"./checkpoints_fr/{TASK}"
        os.makedirs(f"{checkpoint_dir_step_based}/Explicd_only/", exist_ok=True)
        os.makedirs(f"{checkpoint_dir_step_based}/SiT/", exist_ok=True)

        checkpoint_dir_val_score_based=f"./checkoints_fr_val_score_based/{TASK}"
        os.makedirs(f"{checkpoint_dir_val_score_based}/Explicd_only/", exist_ok=True)
        os.makedirs(f"{checkpoint_dir_val_score_based}/SiT/", exist_ok=True)
        #os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != 'None':
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device)
    else:
        encoders, encoder_types, architectures = [None], [None], [None]
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}

    if args.explicd_only==0:

        # model = SiT_models_improved_maybe[args.model](
        #     input_size=latent_size,
        #     num_classes=NUM_OF_CLASSES[TASK],
        #     use_cfg = (args.cfg_prob > 0),
        #     z_dims = z_dims,
        #     encoder_depth=args.encoder_depth,
        #     task = TASK,
        #     **block_kwargs
        # )

        model = SiT_models_with_additional_tokens[args.model](
            input_size=latent_size,
            num_classes=NUM_OF_CLASSES[TASK],
            use_cfg = (args.cfg_prob > 0),
            z_dims = z_dims,
            encoder_depth=args.encoder_depth,
            task = TASK,
            denoise_patches=args.denoise_patches,
            use_actual_latent_of_the_images=args.use_actual_latent,
            trivial_ratio=args.trivial_ratio,
            **block_kwargs
        )

        

        model = model.to(device)
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
        requires_grad(ema, False)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    concept_list = CONCEPTS[TASK]
    conf = argparse.Namespace()
    conf.num_class = NUM_OF_CLASSES[TASK]
    conf.cls_weight = [ 0.076, 0.506, 0.074, 0.137, 0.207]
    conf.dataset = TASK
    conf.data_path = "/home/arsen.abzhanov/Thesis_local/REPA/Explicd/dataset/"
    conf.batch_size = 128
    conf.flag = 2
    conf.do_logits_similarity=DO_LOGITS_SIMILARITY
    conf.new_explicd = args.new_explicd
    conf.trivial_ratio = args.trivial_ratio

    #explicid = ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens(concept_list=concept_list, model_name='biomedclip', config=conf).to(device)
    #explicid = ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Plus_SuperPixels(concept_list=concept_list, model_name='biomedclip', config=conf).to(device)
    #explicid = ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Plus_Representation_Learning(concept_list=concept_list, model_name='biomedclip', config=conf).to(device)
    #explicid = ExpLICD_ViT_L_Classic(concept_list=concept_list, model_name='biomedclip', config=conf).to(device)
    explicid = ExpLICD_ViT_L_Classic_with_Spatial_Bias(concept_list=concept_list, model_name='biomedclip', config=conf).to(device)  
    #explicid = ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Cascade(concept_list=concept_list, model_name='biomedclip', config=conf).to(device)
    #explicid = ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Branching(concept_list=concept_list, model_name='biomedclip', config=conf).to(device)
    #explicid.load_state_dict(torch.load("checkoints_fr_val_score_based/ISIC/Explicd_only/Explicd_additional_tokens_for_sit_starter_better.pt")["explicid"])
    #explicid.cnn = PatchSelectorCNNConscise()
    #explicid.load_state_dict(torch.load("checkoints_fr_val_score_based/ISIC/Explicd_only/Explicd_additional_tokens_for_sit_starter_refined_further_maybe.pt")["explicid"])

    patchifyer_model = Patch_and_Unpatchifier(input_size=latent_size,
            num_classes=NUM_OF_CLASSES[TASK],
            use_cfg = (args.cfg_prob > 0),
            z_dims = z_dims,
            encoder_depth=args.encoder_depth,
            task = TASK,
            **block_kwargs)
    patchifyer_model = patchifyer_model.to(device)
    patchifyer_model.load_state_dict(torch.load("checkpoints_fr/ISIC/SiT/patchifyer_model.pt")["patchifyer_model"])

    explicid_train_transforms = copy.deepcopy(conf.preprocess)
    explicid_train_transforms.transforms.pop(0)
    explicid_train_transforms.transforms.pop(1)
    if explicid.model_name != 'clip':
        explicid_train_transforms.transforms.pop(0)
    
    # (224, 224)
    # (448, 448)
        
    img_size = 448
    #explicid_train_transforms.transforms.insert(0, transforms.CenterCrop(size=(img_size, img_size)))
    explicid_train_transforms.transforms.insert(0, transforms.Resize(size=(img_size,img_size), interpolation=utils.get_interpolation_mode('bicubic'), max_size=None, antialias=True))    
    explicid_train_transforms.transforms.insert(0, transforms.ToPILImage())

    #explicid_train_transforms.transforms.pop(4)
    print("explicid_train_transforms ============",explicid_train_transforms)

    exp_val_transforms = copy.deepcopy(conf.preprocess)
    #print("============",exp_val_transforms)
    exp_val_transforms.transforms.pop(2)
    exp_val_transforms.transforms.insert(0, transforms.ToPILImage())
    exp_val_transforms.transforms.pop(1)
    exp_val_transforms.transforms.insert(1, transforms.Resize(size=(img_size,img_size), interpolation=utils.get_interpolation_mode('bicubic'), max_size=None, antialias=True))
    exp_val_transforms.transforms.pop(2)
    #exp_val_transforms.transforms.pop(4)
    print("exp_val_transforms ============",exp_val_transforms)
    

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        task=TASK,
        do_logits_similarity=DO_LOGITS_SIMILARITY,
        concept_hardness=args.concept_hardness,
        cls_loss_epoch=args.cls_loss_epoch,
    )
    if args.explicd_only==0:
        if accelerator.is_main_process:
            logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.explicd_only==0:
        optimizer = schedulefree.AdamWScheduleFree(list(explicid.parameters()) + list(model.parameters()) + list(patchifyer_model.parameters()), lr=args.learning_rate, warmup_steps=5000 , weight_decay=0.1)
    else:
        optimizer = schedulefree.AdamWScheduleFree(list(explicid.parameters()) + list(patchifyer_model.parameters()), lr=args.learning_rate, warmup_steps=5000 , weight_decay=0.1)

    dual_transform = transforms.Compose([
    DualRandomVerticalFlip(),  # Ensure both images get flipped or not
    DualRandomHorizontalFlip(),
    DualGaussianNoise(p=0.5, mean=0, std=25) if ADD_GAUSSIAN_NOISE else DualGaussianNoise(p=0.0)
])
    dual_val_transform = transforms.Compose([
        #DualRandomGaussianNoise(p=1),
    ])


    if TASK=='ISIC' or TASK=='ISIC_MINE' or TASK=='ISIC_MINIMAL' or TASK=='ISIC_SOFT':
        if DO_MUDDLE_CHECK:
            train_dataset = CustomDataset(args.data_dir, transform= dual_transform, mode='test')
            val_dataset = CustomDataset(args.data_dir, transform= dual_val_transform, mode='val')
            test_dataset = CustomDataset(args.data_dir,transform= None, mode='train')
            #train_muddled_0_025_dataset = CustomDataset(args.data_dir, transform= None, mode='muddled_0_025')
            train_muddled_0_050_dataset = CustomDataset(args.data_dir, transform= None, mode='muddled_0_050')
            #train_muddled_0_075_dataset = CustomDataset(args.data_dir, transform= None, mode='muddled_0_075')
            train_muddled_0_1_dataset = CustomDataset(args.data_dir, transform= None, mode='muddled_0_1')
            train_muddled_0_15_dataset = CustomDataset(args.data_dir, transform= None, mode='muddled_0_15')
            train_muddled_0_2_dataset = CustomDataset(args.data_dir, transform= None, mode='muddled_0_2')

            muddled_datasets = {
                #'0_025': train_muddled_0_025_dataset,
                '0_050': train_muddled_0_050_dataset,
                #'0_075': train_muddled_0_075_dataset,
                '0_1': train_muddled_0_1_dataset,
                '0_15': train_muddled_0_15_dataset,
                '0_2': train_muddled_0_2_dataset
            }

            train_muddled_dataloaders = {
                k: create_muddled_dataloader(v, int(args.batch_size // accelerator.num_processes), args.num_workers)
                for k, v in muddled_datasets.items()
            }

        else:
            train_dataset = CustomDataset(args.data_dir, transform= dual_transform, mode='train')
            val_dataset = CustomDataset(args.data_dir, transform= dual_val_transform, mode='val')
            test_dataset = CustomDataset(args.data_dir,transform= None, mode='test')
    elif TASK=='IDRID':
        train_dataset = CustomDataset(args.data_dir, transform= dual_transform, mode='idrid_train')
        val_dataset = CustomDataset(args.data_dir, transform= dual_val_transform, mode='idrid_val')
        test_dataset = CustomDataset(args.data_dir,transform= None, mode='idrid_test')
    elif TASK=='IDRID_EDEMA':
        train_dataset = CustomDataset(args.data_dir, transform= dual_transform, mode='idrid_edema_train')
        val_dataset = CustomDataset(args.data_dir, transform= dual_val_transform, mode='idrid_edema_val')
        test_dataset = CustomDataset(args.data_dir,transform= None, mode='idrid_edema_test')
    elif TASK=='BUSI':
        train_dataset = CustomDataset(args.data_dir, transform= dual_transform, mode='busi_train')
        val_dataset = CustomDataset(args.data_dir, transform= dual_val_transform, mode='busi_val')
        test_dataset = CustomDataset(args.data_dir,transform= None, mode='busi_test')

    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        #batch_size=local_batch_size,
        batch_size=22,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset train contains {len(train_dataset.image_fnames):,} images ({args.data_dir})")
        logger.info(f"Dataset val contains {len(val_dataset.image_fnames):,} images ({args.data_dir})")
    
    # Prepare models for training:
    if args.explicd_only==0:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode

    explicid.train()
    optimizer.train()
    patchifyer_model.eval()

    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        print(f"ckpt_name {ckpt_name}")
        if args.explicd_only==0:
            checkpoint_resume_locations=f"{checkpoint_dir_step_based}/SiT/{ckpt_name}"
            checkpoint_resume_locations= "checkpoints_fr/ISIC/SiT/hilimsya.pt"
        else:
            checkpoint_resume_locations=f"{checkpoint_dir_step_based}/Explicd_only/{ckpt_name}"
        print(f'loading checkpointed from {checkpoint_resume_locations}')
        ckpt = torch.load(
            f'{checkpoint_resume_locations}',
            map_location='cpu',
            )
        if args.explicd_only==0:
            model.load_state_dict(ckpt['sit'])
            explicid.load_state_dict(ckpt['explicid'])
            ema.load_state_dict(ckpt['ema'])
        else:
            explicid.load_state_dict(ckpt['explicid'])

        optimizer.load_state_dict(ckpt['opt'])
        # To reduce the learning rate, modify the learning rate in the parameter group:
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = param_group['lr'] * 0.01  # Reduce the learning rate by a factor of 0.1
        global_step = ckpt['steps']

    if args.explicd_only==0:
            model = accelerator.prepare(model)
        

    optimizer, train_dataloader, val_dataloader, test_dataloader, explicid, patchifyer_model = accelerator.prepare(
       optimizer, train_dataloader, val_dataloader, test_dataloader, explicid, patchifyer_model
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 36 // accelerator.num_processes
    (_, gt_xs), _ = next(iter(val_dataloader))
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(NUM_OF_CLASSES[TASK], size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
    max_exp_val_f1=0
    max_exp_cnn_critical_val_f1=100
    curve_of_f1 = []
    curve_of_BMAC = []
    val_score_deserves_sampling = False
    if args.explicd_only==1 and not DO_MUDDLE_CHECK:
        print("explicd only")
        optimizer.eval()
        optimizer.zero_grad(set_to_none=True)
        explicid.eval()
        expl_scores, _= validation(explicid, None, test_dataloader, exp_val_transforms, explicd_only=1, accelerator=accelerator)
        print('BMAC: %.5f, f1: %.5f'%(expl_scores["BMAC"], expl_scores["f1"]))
        optimizer.train()
        explicid.train()
    elif not DO_MUDDLE_CHECK:
        print("whole pipeline")
        optimizer.eval()
        optimizer.zero_grad(set_to_none=True)
        explicid.eval()
        model.eval()
        #expl_scores, _= validation(explicid, model, test_dataloader, exp_val_transforms, explicd_only=1)
        #print('BMAC: %.5f, f1: %.5f'%(expl_scores["BMAC"], expl_scores["f1"]))
        optimizer.train()
        explicid.train()
        model.train()

    print(f"new explicd {args.new_explicd}")

    print(f"denoise patches {args.denoise_patches}")

    print(f"use actual latent of the images {args.use_actual_latent}")

    if args.denoise_patches==1 and args.use_actual_latent==0:
        print("this is wrong. you have to chose either denoise patches or use actual latent of the images")
        exit()
    
    #torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    for epoch in range(args.epochs):
        set_seed_mine(42)
        print('Starting epoch {}/{}'.format(epoch+1, args.epochs))
        ##############################################################################################################
        # from samplers import euler_sampler
        # with torch.no_grad():
        #     # Selected elements to be stored
        #     imgs_for_sampling = []
        #     imgs_normalized_for_vae = []
        #     labels_for_sampling = []
        #     # Iterate over the dataset and labels
        #     for img, _ , lab in test_dataset:
        #         img_normalized_for_vae =  img.to(torch.float32) / 127.5 - 1
        #         imgs_for_sampling.append(img)
        #         imgs_normalized_for_vae.append(img_normalized_for_vae)
        #         labels_for_sampling.append(lab)
        #         if len(imgs_for_sampling) == sample_batch_size:
        #             break
        #     latent=vae.encode(torch.stack(imgs_normalized_for_vae, dim=0).to(device))["latent_dist"].mean
        #     #latents_patchified=patchifyer_model(latent)
        #     imgs_for_explicid=prepare__imgs_for_explicid(imgs_for_sampling, exp_val_transforms).to(device)
        #     cls_logits, _, _, _, agg_visual_tokens, _, _, _,  attn_criticial_weights, attn_trivial_weights, vit_l_output, agg_critical_visual_tokens, agg_trivial_visual_tokens, _, _ , critical_mask, trivial_mask  = explicid(imgs_for_explicid)
        #     longer_visual_tokens = torch.cat([agg_critical_visual_tokens, agg_trivial_visual_tokens], dim=1)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Down
            # samples = euler_sampler(
            #     model, 
            #     xT, 
            #     labels_for_sampling,
            #     agg_visual_tokens,
            #     cls_logits,
            #     num_steps=50, 
            #     cfg_scale=0.0,
            #     guidance_low=0.,
            #     guidance_high=1.,
            #     path_type=args.path_type,
            #     heun=False,
            #     attn_critical_weights=attn_criticial_weights, 
            #     attn_trivial_weights=attn_trivial_weights,
            #     longer_visual_tokens = longer_visual_tokens,
            #     vit_l_output=vit_l_output,
            #     critical_mask=critical_mask, 
            #     trivial_mask=trivial_mask,
            #     patchifyer_model=patchifyer_model
            # ).to(torch.float32)
            # recon_samples = vae.decode(samples)["sample"]
            # recon_samples = (recon_samples + 1) / 2.
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Above

            # gt_samples = vae.decode(latent)["sample"]
            # gt_samples = (gt_samples + 1) / 2.
            # gt_samples = accelerator.gather(gt_samples.to(torch.float32))
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Down
            # gt_from_patchifier = vae.decode(latents_patchified)["sample"]
            # gt_from_patchifier = (gt_from_patchifier + 1) / 2.
            # out_samples_pure = accelerator.gather(recon_samples.to(torch.float32))
             # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Above
            #gt_from_patchifier = accelerator.gather(gt_from_patchifier.to(torch.float32))
            ################################################################################ Option Tampering with tokens
            # samples_tampered_list=[]
            # for i in range(8):
            #     if i==0:
            #         longer_visual_tokens_tampered=longer_visual_tokens
            #     else:
            #         longer_visual_tokens_tampered=longer_visual_tokens
            #         longer_visual_tokens_tampered[:,i-1,:]=0
            #     samples_tampered, _, samples_critical_removed = euler_sampler(
            #         model, 
            #         xT, 
            #         torch.stack(labels_for_sampling, dim=0).to(device),
            #         agg_visual_tokens,
            #         cls_logits,
            #         num_steps=50, 
            #         cfg_scale=0.0,
            #         guidance_low=0.,
            #         guidance_high=1.,
            #         path_type=args.path_type,
            #         heun=False,
            #         attn_critical_weights=attn_criticial_weights, 
            #         attn_trivial_weights=attn_trivial_weights,
            #         longer_visual_tokens = longer_visual_tokens_tampered,
            #         vit_l_output=vit_l_output,
            #         critical_mask=critical_mask, 
            #         trivial_mask=trivial_mask,
            #         patchifyer_model=patchifyer_model,
            #         highlight_the_critical_mask=True
            #     )
            #     samples_tampered = vae.decode((samples_tampered -  latents_bias) / latents_scale).sample
            #     samples_tampered = (samples_tampered + 1) / 2.
                
            #     out_samples_tampered = accelerator.gather(samples_tampered.to(torch.float32))
            #     samples_tampered_list.append(out_samples_tampered)

            # accelerator.log({f"samples_untouched": wandb.Image(array2grid(samples_tampered_list[0])),
            #             "gt_samples": wandb.Image(array2grid(gt_samples)),
            #             f"samples first zeroed out": wandb.Image(array2grid(samples_tampered_list[1])),
            #             f"samples second zeroed out": wandb.Image(array2grid(samples_tampered_list[2])),
            #             f"samples third zeroed out": wandb.Image(array2grid(samples_tampered_list[3])),
            #             f"samples fourth zeroed out": wandb.Image(array2grid(samples_tampered_list[4])),
            #             f"samples fifth zeroed out": wandb.Image(array2grid(samples_tampered_list[5])),
            #             f"samples sixth zeroed out": wandb.Image(array2grid(samples_tampered_list[6])),
            #             f"samples seventh zeroed out": wandb.Image(array2grid(samples_tampered_list[7]))
            #             })
            # logging.info("Generating EMA samples done.")
            # return
        #     ################################################################################
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Down
        # accelerator.log({"samples_pure": wandb.Image(array2grid(out_samples_pure)),
        #                     "gt_samples": wandb.Image(array2grid(gt_samples)),
        #                     #"gt_from_patchifier": wandb.Image(array2grid(gt_from_patchifier))
        #                     })
        
        # logging.info("Generating EMA samples done.")
        # return
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Above
        ##############################################################################################################
        if args.explicd_only==0:
            model.train()
        explicid.train()
        optimizer.train()
        #imgs_normalized_for_vae = []
        #     
        for (raw_image, x), y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            with torch.no_grad():
                # raw_image = raw_image.to(torch.float32)/ 255.0  # Convert to [0, 1]
                # for i in range(3):  # Normalize each channel based on ImageNet mean and std
                #     raw_image[..., i] = (raw_image[..., i] - CLIP_DEFAULT_MEAN[i]) / CLIP_DEFAULT_STD[i]

                # # Now scale the image back to [0, 255] but with the distortion already applied
                # raw_image = torch.clip(raw_image * 255, 0, 255)

                imgs_normalized_for_vae =  raw_image.to(torch.float32) / 127.5 - 1
                #imgs_normalized_for_vae.append(img_normalized_for_vae)
                latent=vae.encode(imgs_normalized_for_vae.to(device))["latent_dist"].mean
            # # Convert each tensor to a PIL image and save it
            # for i in range(blurred_raw_images.shape[0]):
            #     blurred_image_tensor = blurred_raw_images[i]  # Shape (3, 256, 256)
            #     rotated_image_tensor = rotated_raw_images[i]
            #     translated_image_tensor = translated_raw_images[i]
            #     color_jittered_image_tensor = color_jittered_raw_images[i]
            #     raw_image_tensor = raw_image[i]
                
            #     # Convert the tensor to a PIL image (needs to be in the form [H, W, C])
            #     blurred_image_pil = Image.fromarray(blurred_image_tensor.cpu().permute(1, 2, 0).numpy())  # Convert from (C, H, W) to (H, W, C)
            #     rotated_image_pil = Image.fromarray(rotated_image_tensor.cpu().permute(1, 2, 0).numpy())
            #     translated_image_pil = Image.fromarray(translated_image_tensor.cpu().permute(1, 2, 0).numpy())
            #     color_jittered_image_pil = Image.fromarray(color_jittered_image_tensor.cpu().permute(1, 2, 0).numpy())
            #     raw_image_tensor_pil = Image.fromarray(raw_image_tensor.cpu().permute(1, 2, 0).numpy())
                
            #     # Save the image
            #     blurred_image_pil.save(os.path.join(imgs_save_dir, f"blurred_image_{i}.png"))
            #     rotated_image_pil.save(os.path.join(imgs_save_dir, f"rotated_image_{i}.png"))
            #     translated_image_pil.save(os.path.join(imgs_save_dir, f"translated_image_{i}.png"))
            #     color_jittered_image_pil.save(os.path.join(imgs_save_dir, f"color_jittered_image_{i}.png"))
            #     raw_image_tensor_pil.save(os.path.join(imgs_save_dir, f"raw_image_{i}.png"))

            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice: 
                # once in train.py and once in sit.py. 
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)
                #exp_test_BMAC, exp_test_acc, _, _ , sit_BMAC, sit_acc= validation(explicid, model, test_dataloader, exp_val_transforms)
            if args.explicd_only==0:
                with accelerator.accumulate(model, explicid, patchifyer_model):
                    imgs_for_explicid=prepare__imgs_for_explicid(raw_image, explicid_train_transforms).to(device)
                    model_kwargs = dict(y=labels)
                    list_of_images=[imgs_for_explicid]
                    if DO_CONTR_LOSS:
                        list_of_images.append(rotation_transform(imgs_for_explicid))
                        list_of_images.append(translation_transform(imgs_for_explicid))
                        list_of_images.append(color_jitter_transform(imgs_for_explicid))
                        list_of_images.append(blur_transform(imgs_for_explicid))

                    # processing_loss, loss, proj_loss, explicid_loss, loss_cls, logits_similarity_loss, _, _, _, _, _, _, attn_explicd_loss, attn_sit_loss, _, cnn_loss_cls= loss_fn(model, latent, latent, model_kwargs, zs=zs, 
                    #                                                     labels=labels, explicid=explicid, explicid_imgs_list=list_of_images, epoch=epoch,  explicd_only=0, do_sit=True, do_pretraining_the_patchifyer=False, patchifyer_model=patchifyer_model)
                    loss_return_dict = loss_fn(model, x, latent, model_kwargs, zs=zs, labels=labels, explicid=explicid, explicid_imgs_list=list_of_images, 
                                               epoch=epoch,  explicd_only=0, do_sit=True, do_pretraining_the_patchifyer=False, patchifyer_model=patchifyer_model, 
                                               denoise_patches=args.denoise_patches, use_actual_latent_of_the_images=args.use_actual_latent)
                    
                    processing_loss = loss_return_dict["processing_loss"]
                    loss = loss_return_dict["denoising_loss"]
                    proj_loss = loss_return_dict["proj_loss"]
                    explicid_loss = loss_return_dict["explicid_loss"]
                    loss_cls = loss_return_dict["loss_cls"]
                    logits_similarity_loss = loss_return_dict["logits_similarity_loss"]
                    attn_explicd_loss = loss_return_dict["attn_explicd_loss"]
                    attn_sit_loss = loss_return_dict["attn_sit_loss"]
                    cnn_loss_cls = loss_return_dict["cnn_loss"]
                    overlap_loss = loss_return_dict["overlap_loss"]
                    loss_cls_with_te = 0*loss_return_dict["loss_cls_with_te"]
                    te_loss = loss_return_dict["te_loss"]

                    processing_loss_mean = processing_loss.mean()
                    loss_mean = loss.mean()
                    proj_loss_mean = proj_loss.mean()* args.proj_coeff
                    explicid_loss_mean = explicid_loss.mean()

                    loss_cls_with_te_mean = loss_cls_with_te.mean()
                    te_loss_mean = te_loss.mean()

                    #explicid_loss_mean = loss_cls.mean()
                    #print("train logits_similarity_loss ", logits_similarity_loss)
                    overlap_loss_mean = 0*overlap_loss.mean()
                    logits_similarity_loss_mean= 0*logits_similarity_loss.mean()
                    if args.new_explicd==0:
                        attn_explicd_loss_mean = attn_explicd_loss.mean() # 10e5
                    elif args.new_explicd==1:
                        attn_explicd_loss_mean = attn_explicd_loss.mean()
                    attn_map_loss_sit_total_mean = 0*attn_sit_loss.mean()
                    cnn_loss_cls_mean = 0*cnn_loss_cls.mean()

                    # if explicid_loss_mean>0.8:
                    #     loss_mean*=0
                        # attn_explicd_loss_mean*=0
                        # attn_map_loss_sit_total_mean*=0

                    # if attn_explicd_loss_mean>0.3:
                    #     loss_mean*=0
                    #     attn_map_loss_sit_total_mean*=0

                    total_loss_magnitude = (processing_loss_mean+loss_mean + proj_loss_mean  + explicid_loss_mean + logits_similarity_loss_mean+
                                            attn_explicd_loss_mean+attn_map_loss_sit_total_mean+cnn_loss_cls_mean+overlap_loss_mean+loss_cls_with_te_mean+te_loss_mean)

                    processing_weight = processing_loss_mean/total_loss_magnitude
                    loss_weight = loss_mean / total_loss_magnitude
                    proj_weight = proj_loss_mean / total_loss_magnitude
                    explicid_weight = explicid_loss_mean / total_loss_magnitude
                    lgs_sim_weight = logits_similarity_loss_mean / total_loss_magnitude
                    attn_explicd_weight = attn_explicd_loss_mean/total_loss_magnitude
                    attn_sit_weight= attn_map_loss_sit_total_mean/total_loss_magnitude
                    cnn_loss_weight = cnn_loss_cls_mean/total_loss_magnitude

                    loss_cls_with_te_mean_weight = loss_cls_with_te_mean/total_loss_magnitude
                    te_loss_mean_weight = te_loss_mean/total_loss_magnitude

                    total_loss = ((processing_weight*processing_loss_mean)+(loss_weight*loss_mean) + (proj_weight*proj_loss_mean) + (explicid_weight*explicid_loss_mean)
                                + (lgs_sim_weight*logits_similarity_loss_mean) + (attn_explicd_weight*logits_similarity_loss_mean) + (attn_sit_weight*attn_map_loss_sit_total_mean)
                                + (cnn_loss_weight*cnn_loss_cls_mean) + (loss_cls_with_te_mean_weight*loss_cls_with_te_mean)+(te_loss_mean_weight*te_loss_mean)) 

                    ## optimization
                    ######################################## Option simple summation of losses
                    accelerator.backward(total_loss_magnitude)
                    ######################################## Option use adaptive loss
                    #accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        #params_to_clip = model.parameters()
                        grad_norm = accelerator.clip_grad_norm_(list(explicid.parameters()) + list(model.parameters()), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        update_ema(ema, model) # change ema function
            else:
                with accelerator.accumulate(explicid, patchifyer_model):
                    imgs_for_explicid=prepare__imgs_for_explicid(raw_image, explicid_train_transforms).to(device)
                    model_kwargs = dict(y=labels)
                    list_of_images=[imgs_for_explicid]
                    if DO_CONTR_LOSS:
                        list_of_images.append(rotation_transform(imgs_for_explicid))
                        list_of_images.append(translation_transform(imgs_for_explicid))
                        list_of_images.append(color_jitter_transform(imgs_for_explicid))
                        list_of_images.append(blur_transform(imgs_for_explicid))

                    # _, _, _, explicid_loss, _, logits_similarity_loss, _, _, _, _, attn_explicd_loss, _, _, cnn_loss_cls = loss_fn(None, latent, latent, model_kwargs, zs=zs, 
                    #                                                     labels=labels, explicid=explicid, explicid_imgs_list=list_of_images, epoch=epoch, explicd_only=1, do_sit=False, do_pretraining_the_patchifyer=False, patchifyer_model=patchifyer_model)
                    loss_return_dict = loss_fn(None, latent, latent, model_kwargs, zs=zs, labels=labels, explicid=explicid, explicid_imgs_list=list_of_images, epoch=epoch, explicd_only=1, do_sit=False, do_pretraining_the_patchifyer=False, patchifyer_model=patchifyer_model)
                    explicid_loss = loss_return_dict["explicid_loss"]
                    logits_similarity_loss = loss_return_dict["logits_similarity_loss"]
                    attn_explicd_loss = loss_return_dict["attn_explicd_loss"]
                    cnn_loss_cls = loss_return_dict["cnn_loss"]
                    overlap_loss = loss_return_dict["overlap_loss"]
                    loss_cls_with_te = loss_return_dict["loss_cls_with_te"]
                    te_loss = 0*loss_return_dict["te_loss"]
                    
                    explicid_loss_mean = explicid_loss.mean()
                    loss_cls_with_te_mean = loss_cls_with_te.mean()
                    te_loss_mean = te_loss.mean()
                    logits_similarity_loss_mean= 0*logits_similarity_loss.mean()
                    if args.new_explicd==0:
                        attn_explicd_loss_mean = 0*attn_explicd_loss.mean() # 10e5
                    elif args.new_explicd==1:
                        attn_explicd_loss_mean = attn_explicd_loss.mean()
                    overlap_loss_mean = overlap_loss.mean()
                    cnn_loss_cls_mean = 0*cnn_loss_cls.mean()
                    accelerator.backward(explicid_loss_mean+logits_similarity_loss_mean+attn_explicd_loss_mean+cnn_loss_cls_mean+overlap_loss_mean+loss_cls_with_te_mean+te_loss_mean)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            ### enter
            #print("enter")
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0 and SAVING_BASED_ON_STEP:
                if accelerator.is_main_process:
                    if args.explicd_only==0:
                        checkpoint = {
                            "sit": model.state_dict(),
                            "explicid": explicid.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": optimizer.state_dict(),
                            "args": args,
                            "steps": global_step,
                        }
                        checkpoint_path = f"{checkpoint_dir_step_based}/SiT/hilimsya.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1000000 or (global_step % args.sampling_steps == 0 and global_step > 0)) and args.explicd_only==0 or val_score_deserves_sampling==True:
                from samplers import euler_sampler
                with torch.no_grad():
                    # Selected elements to be stored
                    imgs_for_sampling = []
                    imgs_normalized_for_vae = []
                    labels_for_sampling = []
                    # Iterate over the dataset and labels
                    for img, _ , lab in test_dataset:
                        img_normalized_for_vae =  img.to(torch.float32) / 127.5 - 1
                        imgs_for_sampling.append(img)
                        imgs_normalized_for_vae.append(img_normalized_for_vae)
                        labels_for_sampling.append(lab)
                        if len(imgs_for_sampling) == sample_batch_size:
                            break
                    latent=vae.encode(torch.stack(imgs_normalized_for_vae, dim=0).to(device))["latent_dist"].mean
                    #latents_patchified=patchifyer_model(latent)
                    imgs_for_explicid=prepare__imgs_for_explicid(imgs_for_sampling, exp_val_transforms).to(device)

                    explicd_return_dict = explicid(imgs_for_explicid)
                    patches = explicd_return_dict["patches"]
                    #patches_colored = explicd_return_dict["patches_colored"]
                    # cls_with_te_logits = explicd_return_dict["cls_with_te_logits"]
                    # cls_logits = explicd_return_dict["cls_logits"]
                    cls_logits = explicd_return_dict["cls_logits"]

                    agg_critical_tokens = explicd_return_dict["agg_critical_visual_tokens_for_SiT"]
                    agg_trivial_tokens = explicd_return_dict["agg_trivial_visual_tokens_for_SiT"]
                    agg_visual_tokens = torch.cat((agg_critical_tokens, agg_trivial_tokens), dim=1)

                    attn_critical_weights = explicd_return_dict["attn_critical_weights"]
                    attn_trivial_weights = explicd_return_dict["attn_trivial_weights"]

                    vit_l_output = explicd_return_dict["vit_l_output"]

                    agg_critical_visual_tokens = explicd_return_dict["agg_critical_visual_tokens"]
                    agg_trivial_visual_tokens = explicd_return_dict["agg_trivial_visual_tokens"]
                    longer_visual_tokens = torch.cat([agg_critical_visual_tokens, agg_trivial_visual_tokens], dim=1)

                    critical_mask = explicd_return_dict["critical_mask"]
                    trivial_mask = explicd_return_dict["trivial_mask"]

                    # num_patches=256
                    # patch_size=14
                    # image_size=224
                    # grid_size = image_size//patch_size

                    # patches = patches.view(sample_batch_size, num_patches, 3, patch_size, patch_size)
                    # patches = patches.view(sample_batch_size, grid_size, grid_size, 3, patch_size, patch_size)
                    # patches = patches.permute(0, 3, 1, 4, 2, 5)
                    # patches = patches.contiguous().view(sample_batch_size, 3, image_size, image_size)

                    images_from_patches = patches_to_images(patches, patch_size=14)
                    images_from_patches = accelerator.gather(images_from_patches.to(torch.float32))

                    #images_from_patches_colored = patches_to_images(patches_colored, patch_size=14)
                    #images_from_patches_colored = accelerator.gather(images_from_patches_colored.to(torch.float32))

                    if args.use_actual_latent==0:
                        gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    elif args.use_actual_latent==1:
                        gt_samples = vae.decode(latent)["sample"]
                    gt_samples = (gt_samples + 1) / 2.
                    gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                    ############################################################################### Option Tampering with tokens
                    samples_tampered_list=[]
                    patches_tampered_list=[]
                    for i in range(NUM_OF_CRITERIA[TASK]+1):
                        if i==0:
                            longer_visual_tokens_tampered=longer_visual_tokens
                        else:
                            longer_visual_tokens_tampered=longer_visual_tokens
                            longer_visual_tokens_tampered[:,i-1,:]=0
                        patches_output, samples_tampered, _, samples_critical_removed = euler_sampler(
                            model, 
                            xT, 
                            torch.stack(labels_for_sampling, dim=0).to(device),
                            agg_visual_tokens,
                            cls_logits,
                            num_steps=50, 
                            cfg_scale=0.0,
                            guidance_low=0.,
                            guidance_high=1.,
                            path_type=args.path_type,
                            heun=False,
                            attn_critical_weights=attn_critical_weights, 
                            attn_trivial_weights=attn_trivial_weights,
                            longer_visual_tokens = longer_visual_tokens_tampered,
                            vit_l_output=vit_l_output,
                            critical_mask=critical_mask, 
                            trivial_mask=trivial_mask,
                            patchifyer_model=patchifyer_model,
                            highlight_the_critical_mask=True,
                            use_actual_latent_of_the_images=args.use_actual_latent
                        )
                        if args.use_actual_latent==0:
                            #samples_tampered = vae.decode((samples_tampered -  latents_bias) / latents_scale).sample
                            samples_tampered = vae.decode(samples_tampered)["sample"]
                        elif args.use_actual_latent==1:
                            samples_tampered = vae.decode(samples_tampered)["sample"]
                        samples_tampered = (samples_tampered + 1) / 2.

                        # patches_output = patches_output.view(sample_batch_size, num_patches, 3, patch_size, patch_size)
                        # patches_output = patches_output.view(sample_batch_size, grid_size, grid_size, 3, patch_size, patch_size)
                        # patches_output = patches_output.permute(0, 3, 1, 4, 2, 5)
                        # patches_output = patches_output.contiguous().view(sample_batch_size, 3, image_size, image_size)
                        # patches_output = accelerator.gather(patches_output.to(torch.float32))
                        
                        if args.denoise_patches==0:
                            out_samples_tampered = accelerator.gather(samples_tampered.to(torch.float32))
                            samples_tampered_list.append(out_samples_tampered)

                        if args.denoise_patches==1:
                            out_images_from_patches_tampered = patches_to_images(patches_output, patch_size=14)
                            patches_tampered_list.append(accelerator.gather(out_images_from_patches_tampered.to(torch.float32)))

                        # out_patches_tampered = accelerator.gather(patches_output.to(torch.float32))
                        # patches_tampered_list.append(out_patches_tampered)


                    if args.denoise_patches==0:
                        accelerator.log({f"samples full package": wandb.Image(array2grid(samples_tampered_list[0])),
                                    "gt_samples": wandb.Image(array2grid(gt_samples)),
                                    f"samples first zeroed out": wandb.Image(array2grid(samples_tampered_list[1])),
                                    f"samples second zeroed out": wandb.Image(array2grid(samples_tampered_list[2])),
                                    f"samples third zeroed out": wandb.Image(array2grid(samples_tampered_list[3])),
                                    f"samples fourth zeroed out": wandb.Image(array2grid(samples_tampered_list[4])),
                                    #f"samples fifth zeroed out": wandb.Image(array2grid(samples_tampered_list[5])),
                                    #f"samples sixth zeroed out": wandb.Image(array2grid(samples_tampered_list[6])),
                                    #f"samples seventh zeroed out": wandb.Image(array2grid(samples_tampered_list[7]))
                                    })
                    elif args.denoise_patches==1:
                        accelerator.log({
                                    f"patches full package": wandb.Image(array2grid(patches_tampered_list[0])),
                                    f"patches ground truth": wandb.Image(array2grid(images_from_patches)),
                                    #f"patches colored": wandb.Image(array2grid(images_from_patches_colored)),
                                    # f"patches first zeroed out": wandb.Image(array2grid(patches_tampered_list[1])),
                                    # f"patches second zeroed out": wandb.Image(array2grid(patches_tampered_list[2])),
                                    # f"patches third zeroed out": wandb.Image(array2grid(patches_tampered_list[3])),
                                    # f"patches fourth zeroed out": wandb.Image(array2grid(patches_tampered_list[4])),
                                    # f"patches fifth zeroed out": wandb.Image(array2grid(patches_tampered_list[5])),
                                    # f"patches sixth zeroed out": wandb.Image(array2grid(patches_tampered_list[6])),
                                    # f"patches seventh zeroed out": wandb.Image(array2grid(patches_tampered_list[7]))
                                    })
                    logging.info("Generating EMA samples done.")
                    if val_score_deserves_sampling:
                        return
            if args.explicd_only==0:
                logs = {
                    #"proc_loss": accelerator.gather(processing_loss_mean).mean().detach().item(),
                    "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                    "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                    "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                    "expl_loss": accelerator.gather(explicid_loss_mean).mean().detach().item(),
                    #"lgs_sim": accelerator.gather(logits_similarity_loss_mean).mean().detach().item(),
                    "attn_exp": accelerator.gather(attn_explicd_loss_mean).mean().detach().item(),
                    #"attn_sit": accelerator.gather(attn_map_loss_sit_total_mean).mean().detach().item(),
                    #"cnn_cls": accelerator.gather(cnn_loss_cls_mean).mean().detach().item(),
                    "overlap_loss": accelerator.gather(overlap_loss_mean).mean().detach().item(),
                    "loss_cls_with_te": accelerator.gather(loss_cls_with_te_mean).mean().detach().item(),
                    "te_loss": accelerator.gather(te_loss_mean).mean().detach().item()
                }

            else:
                logs = {
                    "expl_loss": accelerator.gather(explicid_loss_mean).mean().detach().item(),
                    "attn_exp": accelerator.gather(attn_explicd_loss_mean).mean().detach().item(),
                    "cnn_cls": accelerator.gather(cnn_loss_cls_mean).mean().detach().item(),
                    "lgs_sim": accelerator.gather(logits_similarity_loss_mean).mean().detach().item(),
                    "overlap_loss": accelerator.gather(overlap_loss_mean).mean().detach().item(),
                    "loss_cls_with_te": accelerator.gather(loss_cls_with_te_mean).mean().detach().item(),
                    "te_loss": accelerator.gather(te_loss_mean).mean().detach().item()
                }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break


        explicid.eval()
        optimizer.eval()
        if args.explicd_only==0:
            model.eval()
        exp_pred_list = np.zeros((0), dtype=np.uint8)
        exp_cnn_critical_pred_list = np.zeros((0), dtype=np.uint8)
        gt_list = np.zeros((0), dtype=np.uint8)
        critical_mask_sums = []
        with torch.no_grad():
            for _, ((raw_image, x), y) in enumerate(val_dataloader):
                
                raw_image = raw_image.to(device)
                y = y.to(device)
                z = None
                labels = y

                explicid.eval()
                explicid.zero_grad(set_to_none=True)
                if args.explicd_only==0:
                    model.eval()
                    model.zero_grad(set_to_none=True)
                optimizer.eval()
                torch.cuda.empty_cache()

                imgs_for_explicid=prepare__imgs_for_explicid(raw_image, exp_val_transforms).to(device)
                explicd_return_dict = explicid(imgs_for_explicid)
                # cls_with_te_logits = explicd_return_dict["cls_with_te_logits"]
                # cls_logits = explicd_return_dict["cls_logits"]
                cls_logits = explicd_return_dict["cls_logits"]
                critical_mask = explicd_return_dict["critical_mask"]
                attention_weights = explicd_return_dict["attn_critical_weights"]
                #cnn_logits_critical = explicd_return_dict["cnn_logits_critical"]
                cnn_logits_critical = explicd_return_dict["cnn_logits_critical"]
  
                _, label_pred = torch.max(cls_logits, dim=1)
                _, exp_cnn_critical_label = torch.max(cnn_logits_critical, dim=1)
                if critical_mask is not None:
                    critical_mask_sums.append(critical_mask.sum(dim=(-2,-1)).mean())
                exp_pred_list = np.concatenate((exp_pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0)
                exp_cnn_critical_pred_list = np.concatenate((exp_cnn_critical_pred_list, exp_cnn_critical_label.cpu().numpy().astype(np.uint8)), axis=0)
                gt_list = np.concatenate((gt_list, labels.cpu().numpy().astype(np.uint8)), axis=0)

            imgs_for_explicid = accelerator.gather(imgs_for_explicid.to(torch.float32))
            #print(f"imgs_for_explicid shape {imgs_for_explicid.shape}")

            ###################################
            # B, C, H, W = imgs_for_explicid.shape
            # num_tokens, num_patches = attention_weights.shape[1], attention_weights.shape[2]
            
            # # Assuming the image has been split into non-overlapping patches of size patch_size
            # grid_size = int(np.sqrt(num_patches))  # Assuming square grid of patches
            
            # # Reshape the attention weights to (B, num_tokens, grid_size, grid_size)
            # attention_weights = attention_weights.view(B, num_tokens, grid_size, grid_size)

            # # Scale attention weights to the image grid (224x224)
            # attention_weights_resized = F.interpolate(attention_weights, size=(H, W), mode='bilinear', align_corners=False)
            
            # # The attention weights are now of shape (B, num_tokens, H, W)
            # # Normalize the attention weights to [0, 1]
            # attention_weights_resized = torch.clamp(attention_weights_resized, 0, 1)
            
            # # Now apply the attention weights to the images
            # # Repeat the attention map for each channel (RGB)
            # #imgs_with_attention = imgs_for_explicid.clone()  # Make a copy to modify
            # imgs_with_attention_token_list=[]
            
            # # Multiply each channel of the image with the attention weights (apply it per pixel)
            # for t in range(num_tokens):
            #     # Select attention map for current token, it's of shape (B, H, W)
            #     att_map = attention_weights_resized[:, t]  # (B, H, W)
                
            #     # Repeat the attention map for each channel (to apply it to RGB channels)
            #     att_map_expanded = att_map.unsqueeze(1)  # Shape: (B, 1, H, W)
            #     print(att_map_expanded[0,0,:,:])
                
            #     # Apply the attention map to the image channels
            #     imgs_with_attention_token = 0 * (1 - att_map_expanded) + att_map_expanded * 255

            #     imgs_with_attention_token = accelerator.gather(imgs_with_attention_token.to(torch.float32))
            #     imgs_with_attention_token_list.append(imgs_with_attention_token)
            #     #print(f"imgs_for_explicid shape {imgs_for_explicid.shape}")
                
            # accelerator.log({
            #                 f"imgs_for_explicid": wandb.Image(array2grid(imgs_for_explicid)),
            #                 f"imgs_for_explicid token {0} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[0])),
            #                 f"imgs_for_explicid token {1} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[1])),
            #                 f"imgs_for_explicid token {2} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[2])),
            #                 f"imgs_for_explicid token {3} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[3])),
            #                 f"imgs_for_explicid token {4} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[4])),
            #                 f"imgs_for_explicid token {5} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[5])),
            #                 f"imgs_for_explicid token {6} highlighted": wandb.Image(array2grid(imgs_with_attention_token_list[6]))
            #                  })

            ###################################
            exp_val_BMAC = balanced_accuracy_score(gt_list, exp_pred_list)
            exp_val_correct = np.sum(gt_list == exp_pred_list)
            exp_val_acc = 100 * exp_val_correct / len(exp_pred_list)
            exp_val_f1 = f1_score(gt_list, exp_pred_list, average='macro')
            exp_cnn_critical_val_f1 = f1_score(gt_list, exp_cnn_critical_pred_list, average='macro')

            print(f"Val f1 score {exp_val_f1}")
            if critical_mask is not None:
                print(f"Critical_mask_sums {critical_mask_sums}")
            print(f"Val f1 CNN critical {exp_cnn_critical_val_f1}")

            if args.explicd_only==1:
                if exp_val_f1>max_exp_val_f1 or exp_cnn_critical_val_f1>max_exp_cnn_critical_val_f1:
                    if exp_val_f1>max_exp_val_f1:
                        max_exp_val_f1=exp_val_f1
                    else:
                        max_exp_cnn_critical_val_f1 = exp_cnn_critical_val_f1
                    print('Explicd Val f1', f'{exp_val_f1:.3f}')
                    print('Explicd Val Balanced Acc', f'{exp_val_BMAC:.3f}')
                    if SAVING_BASED_ON_SCORE:
                        checkpoint_val = {
                                "explicid": explicid.state_dict(),
                        }
                        if DO_MUDDLE_CHECK:
                            checkpoint_path = f"{checkpoint_dir_val_score_based}/Explicd_only/Muddled/Explicd_hilimsya_muddled.pt"
                        else:
                            checkpoint_path = f"{checkpoint_dir_val_score_based}/Explicd_only/Explicd_hilimsya.pt"
                        # if args.resume_step == 0:
                        #     torch.save(checkpoint_val, checkpoint_path)
                    explicid.eval()
                    explicid.zero_grad(set_to_none=True)
                    optimizer.eval()
                    expl_scores, tokens_and_gt= validation(explicid, None, test_dataloader, exp_val_transforms, explicd_only=1, accelerator=accelerator)
                    print('Explicd Test f1', f'{expl_scores["f1"]:.3f}')
                    print('Explicd Test Acc', f'{expl_scores["Acc"]:.3f}')
                    print('Explicd Test Balanced Acc', f'{expl_scores["BMAC"]:.3f}')
                    print('Test f1 CNN critical', f'{expl_scores["cnn critical f1"]:.3f}')
                    if epoch>7 and DO_MUDDLE_CHECK and epoch>args.cls_loss_epoch: 
                        #torch.save(tokens_and_gt,"tokens_and_ground_truths/explicd_tokens_and_gts_0")
                        curve_of_f1.clear()
                        curve_of_BMAC.clear()
                        curve_of_f1.append(expl_scores["f1"])
                        curve_of_BMAC.append(expl_scores["BMAC"])
                        for muddle_severity_level in ['0_050','0_1','0_15','0_2']:
                            expl_scores, tokens_and_gt = validation(explicid, None, train_muddled_dataloaders[muddle_severity_level], exp_val_transforms, explicd_only=1, accelerator=accelerator)
                            #torch.save(tokens_and_gt,f"tokens_and_ground_truths/explicd_tokens_and_gts_{muddle_severity_level}")
                            print('Muddle_Severity_Level ', muddle_severity_level)
                            print('Explicd Muddled f1', f'{expl_scores["f1"]:.3f}')
                            print('Explicd Muddled Balanced Acc', f'{expl_scores["BMAC"]:.3f}')
                            curve_of_f1.append(expl_scores["f1"])
                            curve_of_BMAC.append(expl_scores["BMAC"])
                        print('Curve of f1', curve_of_f1)
                        print('Curve of BMAC', curve_of_BMAC)

            else:
                if exp_val_f1>max_exp_val_f1 or exp_cnn_critical_val_f1>max_exp_cnn_critical_val_f1:
                    if exp_val_f1>max_exp_val_f1:
                        max_exp_val_f1=exp_val_f1
                    else:
                        max_exp_cnn_critical_val_f1 = exp_cnn_critical_val_f1
                    print("new best Explicd model")
                    max_exp_val_f1= exp_val_f1
                    print('Explicd Val f1', f'{exp_val_f1:.3f}')
                    print('Explicd Val Balanced Acc', f'{exp_val_BMAC:.3f}')
                    checkpoint = {
                            "sit": model.state_dict(),
                            "explicid": explicid.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": optimizer.state_dict(),
                            "args": args,
                            "steps": global_step,
                        }
                    if DO_MUDDLE_CHECK:
                        checkpoint_path = f"{checkpoint_dir_val_score_based}/SiT/Muddled/SiT_with_explicd_hilimsya_muddled.pt"
                    else:
                        checkpoint_path = f"{checkpoint_dir_val_score_based}/SiT/SiT_with_explicd_hilimsya.pt"
                    if args.resume_step == 0 and SAVING_BASED_ON_SCORE:
                        torch.save(checkpoint_val, checkpoint_path)
                    ####################################################################
                    if max_exp_val_f1>0.94:
                        val_score_deserves_sampling=True
                    explicid.zero_grad(set_to_none=True)
                    model.eval()
                    model.zero_grad(set_to_none=True)
                    optimizer.eval()
                    expl_scores, tokens_and_gt = validation(explicid, model, test_dataloader, exp_val_transforms, explicd_only=0, accelerator=accelerator)
                    print('Explicd Test f1', f'{expl_scores["f1"]:.3f}')
                    print('Explicd Test Acc', f'{expl_scores["Acc"]:.3f}')
                    print('Explicd Test Balanced Acc', f'{expl_scores["BMAC"]:.3f}')
                    print('Test f1 CNN critical', f'{expl_scores["cnn critical f1"]:.3f}')
                    if epoch>7 and DO_MUDDLE_CHECK and epoch>args.cls_loss_epoch:  
                        #torch.save(tokens_and_gt,"tokens_and_ground_truths/sit_tokens_and_gts_0")
                        curve_of_f1.clear()
                        curve_of_BMAC.clear()
                        curve_of_f1.append(expl_scores["f1"])
                        curve_of_BMAC.append(expl_scores["BMAC"])
                        # ['0_025','0_050','0_075','0_1']
                        for muddle_severity_level in ['0_050','0_1','0_15','0_2']:
                            expl_scores, tokens_and_gt = validation(explicid, model, train_muddled_dataloaders[muddle_severity_level], exp_val_transforms, explicd_only=0, accelerator=accelerator)
                            #torch.save(tokens_and_gt,f"tokens_and_ground_truths/sit_tokens_and_gts_{muddle_severity_level}")
                            print('Muddle_Severity_Level ', muddle_severity_level)
                            print('Explicd Muddled f1', f'{expl_scores["f1"]:.3f}')
                            print('Explicd Muddled Balanced Acc', f'{expl_scores["BMAC"]:.3f}')
                            curve_of_f1.append(expl_scores["f1"])
                            curve_of_BMAC.append(expl_scores["BMAC"])
                        print('Curve of f1', curve_of_f1)
                        print('Curve of BMAC', curve_of_BMAC)



        if global_step >= args.max_train_steps:
            break
    if args.explicd_only==0:
        model.eval()  # important! This disables randomized embedding dropout
    optimizer.eval()
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=80000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=96)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=1500)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=2)

    # explicd only
    parser.add_argument("--explicd-only", type=int, default=0)

    # use new explicd method
    parser.add_argument("--new-explicd", type=int, default=1)

    # denoise patches
    parser.add_argument("--denoise-patches", type=int, default=0)

    # use the actual latent of the image
    parser.add_argument("--use-actual-latent", type=int, default=1)

    # task
    parser.add_argument("--task", type=str, choices=['ISIC', 'ISIC_MINE', 'ISIC_MINIMAL', 'ISIC_SOFT', 'IDRID', 'IDRID_EDEMA', 'BUSI', 'BUSI_SOFT'])

    # concept hardness
    parser.add_argument("--concept-hardness", type=str,  choices=["hard","soft_equal","soft_smarter"])

    # cls loss not used intil
    parser.add_argument("--cls-loss_epoch", type=int, default=7)

    # ratio of trivial to critical tokens
    parser.add_argument("--trivial-ratio", type=float, default=1.0)

    # muddle check
    parser.add_argument("--muddle-check", action="store_true")

    # add gaussian noise to transform
    parser.add_argument("--add-gaussian", action="store_true")

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    set_seed_mine(42)
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)