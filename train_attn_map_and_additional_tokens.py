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
from Explicd.model import ExpLICD_ViT_L, ExpLICD, ExpLICD_ViT_L_Multiple_Prompts, ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens, PatchSelectorCNNConscise
from Explicd.concept_dataset import explicid_isic_dict, explicid_isic_dict_mine, explicid_idrid_dict, explicid_idrid_edema_dict, explicid_busi_dict, explicid_busi_soft_smooth_dict
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


CONCEPT_LABEL_MAP_DICT = {
    'ISIC': CONCEPT_LABEL_MAP_ISIC,
    'ISIC_MINE': CONCEPT_LABEL_MAP_ISIC_MINE,

    'IDRID': CONCEPT_LABEL_MAP_IDRID,
    'IDRID_EDEMA': CONCEPT_LABEL_MAP_IDRID_EDEMA,

    'BUSI': CONCEPT_LABEL_MAP_BUSI,
    'BUSI_SOFT': CONCEPT_LABEL_MAP_BUSI_SOFT_SMOOTH
    
}

LIST_OF_TASKS = ['ISIC', 'ISIC_MINE','IDRID', 'IDRID_EDEMA', 'BUSI', 'BUSI_SOFT']

TASK='ISIC'

NUM_OF_CLASSES= {
    'ISIC': 7,
    'ISIC_MINE':7,

    'IDRID': 5,
    'IDRID_EDEMA':3,

    'BUSI': 3,
    'BUSI_SOFT':3
}

CONCEPTS= {
    'ISIC': explicid_isic_dict,
    'ISIC_MINE': explicid_isic_dict_mine,

    'IDRID': explicid_idrid_dict,
    'IDRID_EDEMA': explicid_idrid_edema_dict,

    'BUSI': explicid_busi_dict,
    'BUSI_SOFT': explicid_busi_soft_smooth_dict
}
CONCEPT_HARDNESS_LIST_OPTIONS=["hard","soft_equal","soft_smarter"]

DO_MUDDLE_CHECK=False
ADD_GAUSSIAN_NOISE=False
DO_LOGITS_SIMILARITY=False
CONCEPT_HARDNESS="soft_equal"
DO_CONTR_LOSS = False
noise_levels = [0, 5, 10, 15, 20]

SAVING_BASED_ON_STEP=True
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

CONCEPT_LABEL_MAP = CONCEPT_LABEL_MAP_DICT[TASK]

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
def validation(explicd, model, dataloader, exp_val_transforms, explicd_only=0):
    net = explicd
    if explicd_only==0:
        sit_model = model

    net.eval()
    net.zero_grad(set_to_none=True)
    if explicd_only==0:
        sit_model.eval()
        sit_model.zero_grad(set_to_none=True)

    exp_pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    agg_visual_tokens_list=[]

    with torch.no_grad():
        for _, (raw_image, _, y) in enumerate(dataloader):
            raw_image = raw_image.to("cuda")
            y = y.to("cuda")
            labels = y

            imgs_for_explicid=prepare__imgs_for_explicid(raw_image, exp_val_transforms).to("cuda")
            cls_logits, _, _, _, agg_visual_tokens, _, _, _, _, _, agg_critical_visual_tokens, agg_trivial_visual_tokens, _, _, _= explicd(imgs_for_explicid)
            longer_visual_tokens = torch.cat([agg_critical_visual_tokens, agg_trivial_visual_tokens], dim=1)
            agg_visual_tokens_list.append(longer_visual_tokens)

            _, exp_label_pred = torch.max(cls_logits, dim=1)
            exp_pred_list = np.concatenate((exp_pred_list, exp_label_pred.cpu().numpy().astype(np.uint8)), axis=0)
            gt_list = np.concatenate((gt_list, labels.cpu().numpy().astype(np.uint8)), axis=0)
        

        exp_BMAC = balanced_accuracy_score(gt_list, exp_pred_list)
        exp_correct = np.sum(gt_list == exp_pred_list)
        exp_acc = 100 * exp_correct / len(exp_pred_list)
        exp_val_f1 = f1_score(gt_list, exp_pred_list, average='macro')

        expl_scores={
            "BMAC":exp_BMAC,
            "f1":exp_val_f1,
            "Acc":exp_acc
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
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
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
            **block_kwargs
        )

        

        model = model.to(device)
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
        requires_grad(ema, False)

    concept_list = CONCEPTS[TASK]
    conf = argparse.Namespace()
    conf.num_class = NUM_OF_CLASSES[TASK]
    conf.cls_weight = [ 0.076, 0.506, 0.074, 0.137, 0.207]
    conf.dataset = TASK
    conf.data_path = "/home/arsen.abzhanov/Thesis_local/REPA/Explicd/dataset/"
    conf.batch_size = 128
    conf.flag = 2
    conf.do_logits_similarity=DO_LOGITS_SIMILARITY

    explicid = ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens(concept_list=concept_list, model_name='biomedclip', config=conf).to(device)
    explicid.load_state_dict(torch.load("checkoints_fr_val_score_based/ISIC/Explicd_only/Explicd_additional_tokens_for_sit_starter_better.pt")["explicid"])
    explicid.cnn = PatchSelectorCNNConscise()
    explicid.load_state_dict(torch.load("checkoints_fr_val_score_based/ISIC/Explicd_only/Explicd_additional_tokens_for_sit_starter_refined_further_maybe.pt")["explicid"])

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
    explicid_train_transforms.transforms.insert(0, transforms.CenterCrop(size=(224, 224)))
    explicid_train_transforms.transforms.insert(0, transforms.Resize(size=(224,224), interpolation=utils.get_interpolation_mode('bicubic'), max_size=None, antialias=True))    
    explicid_train_transforms.transforms.insert(0, transforms.ToPILImage())

    print("explicid_train_transforms ============",explicid_train_transforms)

    exp_val_transforms = copy.deepcopy(conf.preprocess)
    #print("============",exp_val_transforms)
    exp_val_transforms.transforms.pop(2)
    exp_val_transforms.transforms.insert(0, transforms.ToPILImage())
    exp_val_transforms.transforms.pop(1)
    exp_val_transforms.transforms.insert(1, transforms.Resize(size=(224,224), interpolation=utils.get_interpolation_mode('bicubic'), max_size=None, antialias=True))

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
        concept_hardness=CONCEPT_HARDNESS
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


    if TASK=='ISIC' or TASK=='ISIC_MINE':
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
        val_dataset = CustomDataset(args.data_dir, mode='idrid_val')
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
        batch_size=local_batch_size,
        shuffle=False,
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
    sample_batch_size = 64 // accelerator.num_processes
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
    curve_of_f1 = []
    curve_of_BMAC = []
    if args.explicd_only==1:
        print("explicd only")
        optimizer.eval()
        optimizer.zero_grad(set_to_none=True)
        explicid.eval()
        expl_scores, _= validation(explicid, None, test_dataloader, exp_val_transforms, explicd_only=1)
        print('BMAC: %.5f, f1: %.5f'%(expl_scores["BMAC"], expl_scores["f1"]))
        optimizer.train()
        explicid.train()
    else:
        print("whole pipeline")
        optimizer.eval()
        optimizer.zero_grad(set_to_none=True)
        explicid.eval()
        model.eval()
        expl_scores, _= validation(explicid, model, test_dataloader, exp_val_transforms, explicd_only=1)
        print('BMAC: %.5f, f1: %.5f'%(expl_scores["BMAC"], expl_scores["f1"]))
        optimizer.train()
        explicid.train()
        model.train()
    
    
    for epoch in range(args.epochs):
        set_seed_mine(42)
        print('Starting epoch {}/{}'.format(epoch+1, args.epochs))
        ##############################################################################################################
        # from samplers import euler_sampler
        # with torch.no_grad():
        #     # Selected elements to be stored
        #     imgs_for_sampling = []
        #     imgs_normalized_for_vae = []
        #     # Iterate over the dataset and labels
        #     for img, _ , lab in test_dataset:
        #         img_normalized_for_vae =  img.to(torch.float32) / 127.5 - 1
        #         imgs_for_sampling.append(img)
        #         imgs_normalized_for_vae.append(img_normalized_for_vae)
        #         if len(imgs_for_sampling) == sample_batch_size:
        #             break
        #     latent=vae.encode(torch.stack(imgs_normalized_for_vae, dim=0).to(device))["latent_dist"].mean
        #     #latents_patchified=patchifyer_model(latent)
        #     imgs_for_explicid=prepare__imgs_for_explicid(imgs_for_sampling, exp_val_transforms).to(device)
        #     cls_logits, _, _, _, agg_visual_tokens, _, _, attn_criticial_weights, attn_trivial_weights, vit_l_output, agg_critical_visual_tokens, agg_trivial_visual_tokens, _, critical_mask, trivial_mask  = explicid(imgs_for_explicid)
        #     longer_visual_tokens = torch.cat([agg_critical_visual_tokens, agg_trivial_visual_tokens], dim=1)
        #     samples = euler_sampler(
        #         model, 
        #         xT, 
        #         ys,
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
        #         longer_visual_tokens = longer_visual_tokens,
        #         vit_l_output=vit_l_output,
        #         critical_mask=critical_mask, 
        #         trivial_mask=trivial_mask,
        #         patchifyer_model=patchifyer_model
        #     ).to(torch.float32)
        #     recon_samples = vae.decode(samples)["sample"]
        #     recon_samples = (recon_samples + 1) / 2.

        #     gt_samples = vae.decode(latent)["sample"]
        #     gt_samples = (gt_samples + 1) / 2.
            
        #     # gt_from_patchifier = vae.decode(latents_patchified)["sample"]
        #     # gt_from_patchifier = (gt_from_patchifier + 1) / 2.

        #     out_samples_pure = accelerator.gather(recon_samples.to(torch.float32))
        #     gt_samples = accelerator.gather(gt_samples.to(torch.float32))
        #     gt_from_patchifier = accelerator.gather(gt_from_patchifier.to(torch.float32))
        #     ################################################################################ Option Tampering with tokens
        #     # samples_tampered_list=[]
        #     # for i in range(8):
        #     #     if i==0:
        #     #         longer_visual_tokens_tampered=longer_visual_tokens
        #     #     else:
        #     #         longer_visual_tokens_tampered=longer_visual_tokens[:,i-1,:]=0
        #     #     samples_tampered = euler_sampler(
        #     #         model, 
        #     #         xT, 
        #     #         ys,
        #     #         agg_visual_tokens,
        #     #         cls_logits,
        #     #         num_steps=50, 
        #     #         cfg_scale=4.0,
        #     #         guidance_low=0.,
        #     #         guidance_high=1.,
        #     #         path_type=args.path_type,
        #     #         heun=False,
        #     #         attn_critical_weights=attn_criticial_weights, 
        #     #         attn_trivial_weights=attn_trivial_weights,
        #     #         longer_visual_tokens = longer_visual_tokens_tampered,
        #     #         vit_l_output=vit_l_output,
        #     #         critical_mask=critical_mask, 
        #     #         trivial_mask=trivial_mask,
        #     #         patchifyer_model=patchifyer_model
        #     #     ).to(torch.float32)
        #     #     samples_tampered = vae.decode((samples_tampered -  latents_bias) / latents_scale).sample
        #     #     samples_tampered = (samples_tampered + 1) / 2.
                
        #     #     out_samples_tampered = accelerator.gather(samples_tampered.to(torch.float32))
        #     #     samples_tampered_list.append(out_samples_tampered)

        #     # accelerator.log({f"samples_untouched": wandb.Image(array2grid(samples_tampered_list[0])),
        #     #             "gt_samples": wandb.Image(array2grid(gt_samples)),
        #     #             f"samples first zeroed out": wandb.Image(array2grid(samples_tampered_list[1])),
        #     #             f"samples second zeroed out": wandb.Image(array2grid(samples_tampered_list[2])),
        #     #             f"samples third zeroed out": wandb.Image(array2grid(samples_tampered_list[3])),
        #     #             f"samples fourth zeroed out": wandb.Image(array2grid(samples_tampered_list[4])),
        #     #             f"samples fifth zeroed out": wandb.Image(array2grid(samples_tampered_list[5])),
        #     #             f"samples sixth zeroed out": wandb.Image(array2grid(samples_tampered_list[6])),
        #     #             f"samples seventh zeroed out": wandb.Image(array2grid(samples_tampered_list[7]))
        #     #             })
        #     # logging.info("Generating EMA samples done.")
        #     # return
        #     ################################################################################
        
        # accelerator.log({"samples_pure": wandb.Image(array2grid(out_samples_pure)),
        #                     "gt_samples": wandb.Image(array2grid(gt_samples)),
        #                     #"gt_from_patchifier": wandb.Image(array2grid(gt_from_patchifier))
        #                     })
        
        # logging.info("Generating EMA samples done.")
        # return
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

                    processing_loss, loss, proj_loss, explicid_loss, _, logits_similarity_loss, _, _, _, _, _, _, attn_explicd_loss, attn_map_loss_sit_total, _, cnn_loss_cls= loss_fn(model, latent, latent, model_kwargs, zs=zs, 
                                                                        labels=labels, explicid=explicid, explicid_imgs_list=list_of_images, epoch=epoch,  explicd_only=0, do_sit=True, do_pretraining_the_patchifyer=False, patchifyer_model=patchifyer_model)
                    processing_loss_mean = processing_loss.mean()
                    loss_mean = loss.mean()
                    proj_loss_mean = proj_loss.mean()* args.proj_coeff
                    explicid_loss_mean = explicid_loss.mean() *2
                    logits_similarity_loss_mean= logits_similarity_loss.mean()
                    attn_explicd_loss_mean = attn_explicd_loss.mean()
                    attn_map_loss_sit_total_mean = attn_map_loss_sit_total.mean()
                    cnn_loss_cls_mean = cnn_loss_cls.mean()

                    total_loss_magnitude = processing_loss_mean+loss_mean + proj_loss_mean  + explicid_loss_mean + logits_similarity_loss_mean+attn_explicd_loss_mean+attn_map_loss_sit_total_mean+cnn_loss_cls_mean

                    processing_weight = processing_loss_mean/total_loss_magnitude
                    loss_weight = loss_mean / total_loss_magnitude
                    proj_weight = proj_loss_mean / total_loss_magnitude
                    explicid_weight = explicid_loss_mean / total_loss_magnitude
                    lgs_sim_weight = logits_similarity_loss_mean / total_loss_magnitude
                    attn_explicd_weight = attn_explicd_loss_mean/total_loss_magnitude
                    attn_sit_weight= attn_map_loss_sit_total_mean/total_loss_magnitude
                    cnn_loss_weight = cnn_loss_cls_mean/total_loss_magnitude

                    total_loss = ((processing_weight*processing_loss_mean)+(loss_weight*loss_mean) + (proj_weight*proj_loss_mean) + (explicid_weight*explicid_loss_mean)
                                + (lgs_sim_weight*logits_similarity_loss_mean) + (attn_explicd_weight*logits_similarity_loss_mean) + (attn_sit_weight*attn_map_loss_sit_total_mean)
                                + (cnn_loss_weight*cnn_loss_cls_mean)) 

                    ## optimization
                    accelerator.backward(total_loss)
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

                    _, _, _, explicid_loss, _, logits_similarity_loss, _, _, _, _, attn_explicd_loss, _, _, cnn_loss_cls = loss_fn(None, x, model_kwargs, zs=zs, 
                                                                        labels=labels, explicid=explicid, explicid_imgs_list=list_of_images, epoch=epoch, explicd_only=1)

                    explicid_loss_mean = explicid_loss.mean()
                    logits_similarity_loss_mean= logits_similarity_loss.mean()
                    attn_explicd_loss_mean = attn_explicd_loss.mean()
                    cnn_loss_cls_mean = cnn_loss_cls.mean()
                    accelerator.backward(explicid_loss_mean+logits_similarity_loss_mean+attn_explicd_loss_mean+cnn_loss_cls_mean)
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

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)) and args.explicd_only==0:
                from samplers import euler_sampler
                with torch.no_grad():
                    # Selected elements to be stored
                    imgs_for_sampling = []
                    imgs_normalized_for_vae = []
                    # Iterate over the dataset and labels
                    for img, _ , lab in test_dataset:
                        img_normalized_for_vae =  img.to(torch.float32) / 127.5 - 1
                        imgs_for_sampling.append(img)
                        imgs_normalized_for_vae.append(img_normalized_for_vae)
                        if len(imgs_for_sampling) == sample_batch_size:
                            break

                    latent_sampling=vae.encode(torch.stack(imgs_normalized_for_vae, dim=0).to(device))["latent_dist"].mean
                    imgs_for_explicid=prepare__imgs_for_explicid(imgs_for_sampling, exp_val_transforms).to(device)
                    cls_logits, _, _, _, agg_visual_tokens, _, _, attn_criticial_weights, attn_trivial_weights, vit_l_output, agg_critical_visual_tokens, agg_trivial_visual_tokens, _, critical_mask, trivial_mask  = explicid(imgs_for_explicid)
                    longer_visual_tokens = torch.cat([agg_critical_visual_tokens, agg_trivial_visual_tokens], dim=1)
                    samples = euler_sampler(
                            model, 
                            latent_sampling, 
                            ys,
                            agg_visual_tokens,
                            cls_logits,
                            num_steps=50, 
                            cfg_scale=0.0,
                            guidance_low=0.,
                            guidance_high=1.,
                            path_type=args.path_type,
                            heun=False,
                            attn_critical_weights=attn_criticial_weights, 
                            attn_trivial_weights=attn_trivial_weights,
                            longer_visual_tokens = longer_visual_tokens,
                            vit_l_output=vit_l_output,
                            critical_mask=critical_mask, 
                            trivial_mask=trivial_mask,
                            patchifyer_model=patchifyer_model
                        ).to(torch.float32)
                    samples = vae.decode(samples)["sample"]
                    samples = (samples + 1) / 2.

                    gt_samples = vae.decode(latent)["sample"]
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")
            if args.explicd_only==0:
                logs = {
                    "proc_loss": accelerator.gather(processing_loss_mean).mean().detach().item(),
                    "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                    "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                    "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                    "expl_loss": accelerator.gather(explicid_loss_mean).mean().detach().item(),
                    "lgs_sim": accelerator.gather(logits_similarity_loss_mean).mean().detach().item(),
                    "attn_exp": accelerator.gather(attn_explicd_loss_mean).mean().detach().item(),
                    "attn_sit": accelerator.gather(attn_map_loss_sit_total_mean).mean().detach().item(),
                    "cnn_cls": accelerator.gather(cnn_loss_cls_mean).mean().detach().item(),
                }

            else:
                logs = {
                    "expl_loss": accelerator.gather(explicid_loss_mean).mean().detach().item(),
                    "attn_exp": accelerator.gather(attn_explicd_loss_mean).mean().detach().item(),
                    "cnn_cls": accelerator.gather(cnn_loss_cls_mean).mean().detach().item(),
                    "lgs_sim": accelerator.gather(logits_similarity_loss_mean).mean().detach().item(),
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
        gt_list = np.zeros((0), dtype=np.uint8)
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
                cls_logits, _, _, _, agg_visual_tokens, _, _, _, _, _, _, _, _, _, _ = explicid(imgs_for_explicid)
                _, label_pred = torch.max(cls_logits, dim=1)

                exp_pred_list = np.concatenate((exp_pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0)
                gt_list = np.concatenate((gt_list, labels.cpu().numpy().astype(np.uint8)), axis=0)


            
            exp_val_BMAC = balanced_accuracy_score(gt_list, exp_pred_list)
            exp_val_correct = np.sum(gt_list == exp_pred_list)
            exp_val_acc = 100 * exp_val_correct / len(exp_pred_list)
            exp_val_f1 = f1_score(gt_list, exp_pred_list, average='macro')

            if args.explicd_only==1:
                if exp_val_f1>max_exp_val_f1:
                    max_exp_val_f1=exp_val_f1
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
                        if args.resume_step == 0:
                            torch.save(checkpoint_val, checkpoint_path)
                    explicid.eval()
                    explicid.zero_grad(set_to_none=True)
                    optimizer.eval()
                    expl_scores, tokens_and_gt= validation(explicid, None, test_dataloader, exp_val_transforms, explicd_only=1)
                    print('Explicd Test f1', f'{expl_scores["f1"]:.3f}')
                    print('Explicd Test Acc', f'{expl_scores["Acc"]:.3f}')
                    print('Explicd Test Balanced Acc', f'{expl_scores["BMAC"]:.3f}')
                    if epoch>7 and DO_MUDDLE_CHECK: 
                        torch.save(tokens_and_gt,"tokens_and_ground_truths/explicd_tokens_and_gts_0")
                        curve_of_f1.clear()
                        curve_of_BMAC.clear()
                        curve_of_f1.append(expl_scores["f1"])
                        curve_of_BMAC.append(expl_scores["BMAC"])
                        for muddle_severity_level in ['0_050','0_1','0_15','0_2']:
                            expl_scores, _, _, tokens_and_gt = validation(explicid, None, train_muddled_dataloaders[muddle_severity_level], exp_val_transforms, explicd_only=1)
                            torch.save(tokens_and_gt,f"tokens_and_ground_truths/explicd_tokens_and_gts_{muddle_severity_level}")
                            print('Muddle_Severity_Level ', muddle_severity_level)
                            print('Explicd Muddled f1', f'{expl_scores["f1"]:.3f}')
                            print('Explicd Muddled Balanced Acc', f'{expl_scores["BMAC"]:.3f}')
                            curve_of_f1.append(expl_scores["f1"])
                            curve_of_BMAC.append(expl_scores["BMAC"])
                        print('Curve of f1', curve_of_f1)
                        print('Curve of BMAC', curve_of_BMAC)

            else:
                if exp_val_f1>max_exp_val_f1:
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
                    explicid.zero_grad(set_to_none=True)
                    model.eval()
                    model.zero_grad(set_to_none=True)
                    optimizer.eval()
                    expl_scores, sit_scores, expl_refined_scores, tokens_and_gt = validation(explicid, model, test_dataloader, exp_val_transforms, explicd_only=0)
                    print('Explicd Test f1', f'{expl_scores["f1"]:.3f}')
                    print('Explicd Test Acc', f'{expl_scores["Acc"]:.3f}')
                    print('Explicd Test Balanced Acc', f'{expl_scores["BMAC"]:.3f}')
                    if epoch>7 and DO_MUDDLE_CHECK:  
                        torch.save(tokens_and_gt,"tokens_and_ground_truths/sit_tokens_and_gts_0")
                        curve_of_f1.clear()
                        curve_of_BMAC.clear()
                        curve_of_f1.append(expl_scores["f1"])
                        curve_of_BMAC.append(expl_scores["BMAC"])
                        # ['0_025','0_050','0_075','0_1']
                        for muddle_severity_level in ['0_050','0_1','0_15','0_2']:
                            expl_scores, sit_scores, expl_refined_scores, tokens_and_gt = validation(explicid, model, train_muddled_dataloaders[muddle_severity_level], exp_val_transforms, explicd_only=0)
                            torch.save(tokens_and_gt,f"tokens_and_ground_truths/sit_tokens_and_gts_{muddle_severity_level}")
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
    parser.add_argument("--sampling-steps", type=int, default=5000)
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
    parser.add_argument("--checkpointing-steps", type=int, default=5000)
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