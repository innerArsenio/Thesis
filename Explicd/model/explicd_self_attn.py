import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer
from torchvision import transforms
from .utils import FFN

from transformers import ViTModel, ViTConfig

import pdb

import clip
import random
import torch
import torch.nn as nn
# import cv2
import matplotlib.pyplot as plt
from skimage import measure
from timm import create_model
from skimage.segmentation import slic
from skimage.color import rgb2lab
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from skimage import img_as_ubyte
import skimage
from skimage.color import label2rgb
import torchvision.transforms as T
from skimage.util import img_as_float
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import copy
import entropy_lens as te

NUM_OF_CRITERIA = {
    'ISIC': 7,
    'ISIC_MINE': 6,
    'ISIC_MINIMAL': 7,
    'ISIC_SOFT': 7,

    'IDRID': 5,
    'IDRID_SOFT':7,
    'IDRID_EDEMA': 6,
    'IDRID_EDEMA_SOFT':4,

    'BUSI': 6,
    'BUSI_SOFT': 6
}

NUM_OF_SIMILARITIES = {
    'ISIC': 34,
    'ISIC_MINE': 24,
    'ISIC_MINIMAL': 49,
    'ISIC_SOFT': 34,

    'IDRID': 18,
    'IDRID_SOFT': 30,
    'IDRID_EDEMA': 15,
    'IDRID_EDEMA_SOFT': 12,
    
    'BUSI': 17,
    'BUSI_SOFT': 18
}

def get_prefix(task: str, key: str) -> str:
    if task== "ISIC" or task == "ISIC_MINE" or task == "ISIC_MINIMAL" or task == "ISIC_SOFT":
        return f"this is a dermoscopic image, the {key} of the lesion is "
    elif task == "IDRID" or task=='IDRID_EDEMA' or task == "IDRID_SOFT":
        return f"this is a fundus image, the {key} of the eye is "
    elif task == "IDRID_SOFT" or task == "IDRID_EDEMA_SOFT":
        return f"this is a fundus image, there are "
    elif task == "BUSI" or task == "BUSI_SOFT":
        return f"this is an ultrasound image of human breasts, the {key} of the tissue is "
    else:
        raise ValueError(f"Unknown task: {task}. Must be either 'derm' or 'us'")
 
class SpatialAwarePatchLoss_mine(nn.Module):
    def __init__(self, gamma=5.0, sigma=3):
        """
        Args:
            gamma (float): Controls how strongly distance affects penalties.
            sigma (float): Controls the sharpness of spatial weighting.
        """
        super().__init__()
        self.gamma = gamma
        self.sigma = sigma

    def compute_distance_matrix(self, H, W, device, sigma=1):
        """Compute normalized spatial distance matrix for a grid (H, W)."""
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1)  # (H, W, 2)
        coords = coords.reshape(-1, 2).to(device)  # (256, 2) for a 16x16 grid

        # Compute pairwise Euclidean distance
        dist_matrix = torch.cdist(coords.float(), coords.float(), p=2)  # (256, 256)

        # Normalize to [0,1] using the largest distance in the grid
        #dist_matrix /= dist_matrix.max()
        return dist_matrix  # (256, 256)

    def compute_self_similarity(self, feats):
        """
        Compute cosine similarity between all spatial patches.
        feats: (B, C, 16, 16) -> Flatten to (B, C, 256)
        Returns: Similarity matrix (B, 256, 256)
        """
        B, C, H, W = feats.shape
        feats = feats.view(B, C, H * W)  # (B, C, 256)
        feats = F.normalize(feats, dim=1)  # Normalize along feature dim
        sim_matrix = torch.bmm(feats.transpose(1, 2), feats)  # (B, 256, 256)
        return sim_matrix

    def weighted_kl_div(self, attn_sim, feat_sim, W_spatial):
        """
        Compute KL divergence weighted by spatial similarity.
        """
        B, N, N = attn_sim.shape  # (B, 256, 256)
        
        #attn_sim = F.normalize(attn_sim, p=1, dim=-1)  # Normalize rows to sum to 1
        #feat_sim = F.normalize(feat_sim, p=1, dim=-1)  # Normalize rows to sum to 1
        
        kl_div = F.kl_div(attn_sim.log(), feat_sim, reduction='none')  # (B, 256, 256)
        # print(kl_div)
        # print(W_spatial)
        
        # Weight by spatial proximity
        weighted_kl = kl_div * W_spatial.unsqueeze(0).to(attn_sim.device)  # (B, 256, 256)
        weighted_kl = torch.nan_to_num(weighted_kl).to(attn_sim.device)
        #print(weighted_kl.mean())
        
        return 100*weighted_kl.mean()  # Scalar loss
    
    def fuzzy_hue_similarity(self, h1, h2, sigma=15):
        """
        Computes fuzzy similarity for hue (handles wrap-around at 0/360 degrees).
        """
        diff = torch.abs(h1 - h2)
        diff = torch.minimum(diff, 360 - diff)  # Handle hue wrap-around
        return torch.exp(-diff ** 2 / (2 * sigma ** 2))  # Gaussian membership
    
    def fuzzy_saturation_similarity(self, s1, s2, sigma=0.2):
        return torch.exp(-torch.abs(s1 - s2) / sigma)

    def fuzzy_value_similarity(self, v1, v2, sigma=0.2):
        return torch.exp(-torch.abs(v1 - v2) / sigma)

    def fuzzy_patch_similarity(self, h1, s1, v1, h2, s2, v2):
        """
        Compute fuzzy similarity between patches using HSV features.
        """
        hue_sim = self.fuzzy_hue_similarity(h1, h2)
        sat_sim = self.fuzzy_saturation_similarity(s1, s2)
        val_sim = self.fuzzy_value_similarity(v1, v2)
        
        # Combine fuzzy rules (weighted sum)
        return 0.5 * hue_sim + 0.3 * sat_sim + 0.2 * val_sim

    def forward(self, image_feats, attn_critical, attn_non_critical):
        """
        Args:
            image_feats (torch.Tensor): Image features (B, 1024, 16, 16).
            mask (torch.Tensor): Binary segmentation mask (B, 1, 16, 16).

        Returns:
            torch.Tensor: Spatially aware loss.
        """
        B, C, H, W = image_feats.shape
        _, num_critical, _ = attn_critical.shape
        _, num_trivial, _ = attn_non_critical.shape
        num_patches = H * W
        device = image_feats.device

        # Compute feature similarity
        feat_sim = self.compute_self_similarity(image_feats)

        # Flatten patches (B, 1024, 256) and mask (B, 1, 256)
        image_feats = image_feats.view(B, C, num_patches)
        attn_critical = attn_critical.view(B, num_critical, num_patches)
        attn_non_critical = attn_non_critical.view(B, num_trivial, num_patches)
        #mask_coverage = mask.mean(dim=2, keepdim=True)  # (B, 1, 1)

        #lambda_within = torch.exp(-self.gamma * mask_coverage)
        #lambda_between = torch.exp(self.gamma * mask_coverage)
        lambda_within = 1
        lambda_between = 1

        # Compute spatial distance matrix (256, 256)
        D = self.compute_distance_matrix(H, W, device).to(device)  # (256, 256)

        # Convert distance into weight (closer patches get higher weights)
        #spatial_weight = torch.exp(-self.gamma * D)  # (256, 256)
        spatial_weight = torch.exp(- (D ** 2) / (2 * self.sigma ** 2))
        feat_sim = feat_sim * spatial_weight[None, :, :].to(device)  # Shape: (B, 256, 256)
        feat_sim = F.normalize(feat_sim, p=1, dim=-1) + 1e-6
        total_loss=0

        ############################################# Option use fuzzy similarity
        # Compute pairwise fuzzy similarity
        feat_sim = torch.zeros(B, 256, 256, device=device)
        h1 = image_feats[:, 0, :].unsqueeze(2)  # (B, 256, 1)
        h2 = image_feats[:, 0, :].unsqueeze(1)  # (B, 1, 256)
        diff = torch.abs(h1 - h2)  # (B, 256, 256)
        diff = torch.minimum(diff, 360 - diff)  # Handle hue wrap-around
        hue_sim = torch.exp(-diff ** 2 / (2 * 15 ** 2))  # (B, 256, 256)
        s1, s2 = image_feats[:, 1, :].unsqueeze(2), image_feats[:, 1, :].unsqueeze(1)
        v1, v2 = image_feats[:, 2, :].unsqueeze(2), image_feats[:, 2, :].unsqueeze(1)

        sat_sim = torch.exp(-torch.abs(s1 - s2) / 0.2)  # (B, 256, 256)
        val_sim = torch.exp(-torch.abs(v1 - v2) / 0.2)  # (B, 256, 256)
        feat_sim = 0.7 * hue_sim + 0.2 * sat_sim + 0.1 * val_sim  # (B, 256, 256)
        feat_sim = feat_sim * spatial_weight[None, :, :].to(device)  # Shape: (B, 256, 256)  Option to use spatial weights
        # feat_sim = feat_sim + 1e-6  # Prevent log(0)
        # feat_sim = feat_sim / feat_sim.sum(dim=-1, keepdim=True)  # Normalize

        #############################################
                
        for i in range(num_critical):
            # if i in [1,4,5,6]:
            #     continue

            if i!=0:
                continue
            ########################################## Option use both types of attention weights
            # # Extract critical and trivial patches
            # critical_patches = image_feats * attn_critical[:,i, :].unsqueeze(1)  # (B, 1024, 256) * (B, 1, 256)
            # trivial_patches = image_feats * attn_non_critical[:,i, :].unsqueeze(1)  # (B, 1024, 256) * (B, 1, 256)

            # # Normalize features for cosine similarity
            # critical_patches = F.normalize(critical_patches, p=2, dim=1)
            # trivial_patches = F.normalize(trivial_patches, p=2, dim=1)

            # # Compute similarity matrices
            # similarity_critical = torch.bmm(critical_patches.transpose(1, 2), critical_patches)  # (B, 256, 256)
            # similarity_between = torch.bmm(critical_patches.transpose(1, 2), trivial_patches)  # (B, 256, 256)

            # # Loss 1: Critical patches should be similar if spatially close
            # loss_within = (((similarity_critical - 1) ** 2 * spatial_weight)*lambda_within).mean()

            # # Loss 2: Trivial patches should be dissimilar to nearby critical patches
            # loss_between = ((similarity_between ** 2 * spatial_weight)*lambda_between).mean()

            # # Total loss
            # total_loss += loss_within + loss_between
            ########################################## Option to use only the critical weights
            attn_sim_crit = torch.bmm(attn_critical[:,i, :].unsqueeze(2), attn_critical[:,i, :].unsqueeze(1))  # (B, 256, 256)
            #attn_sim_crit = attn_sim_crit / (attn_sim_crit.max(dim=-1, keepdim=True).values)  # Normalize scale
            #attn_sim_crit = F.normalize(attn_sim_crit, p=1, dim=-1) + 1e-6

            attn_sim_trivial = torch.bmm(attn_non_critical[:,i, :].unsqueeze(2), attn_non_critical[:,i, :].unsqueeze(1))  # (B, 256, 256)
            #attn_sim_trivial = attn_sim_trivial / (attn_sim_trivial.max(dim=-1, keepdim=True).values)  # Normalize scale
            # Self-similarity loss
            loss_sim = F.mse_loss(attn_sim_crit, feat_sim)
            #loss_sim = F.mse_loss(attn_sim_crit, feat_sim) + F.mse_loss(attn_sim_trivial, feat_sim)
            #loss_sim = torch.mean(spatial_weight * F.mse_loss(attn_sim_crit, feat_sim))
            
            
            #loss_sim = F.kl_div(attn_sim_crit.log(), feat_sim, reduction='batchmean')
            #loss_sim = self.weighted_kl_div(attn_sim_crit, feat_sim, spatial_weight)
            #loss_sim = torch.mean(spatial_weight * F.cosine_similarity(attn_sim_crit, feat_sim, dim=-1))
            ########################################### Option use Huber Loss
            # diff = attn_sim_crit - feat_sim  # Difference matrix (B, 256, 256)
            # delta=0.1
            # abs_diff = torch.abs(diff)
            # loss = torch.where(abs_diff < delta,  # Huber loss formulation
            #                 0.5 * abs_diff ** 2,
            #                 delta * (abs_diff - 0.5 * delta))
            
            # # Weight by spatial similarity
            # weighted_loss = loss * spatial_weight[None, :, :].to(attn_sim_crit.device)
            # loss_sim = weighted_loss.mean()  # Final loss scalar
            ###########################################
            # Smoothness loss
            loss_smooth = torch.mean(spatial_weight * (attn_critical[:,i, :].unsqueeze(2) - attn_critical[:,i, :].unsqueeze(1)) ** 2)
            #total_loss += 10*loss_sim + loss_smooth

            
            total_loss += 10*loss_sim
            #total_loss += F.mse_loss(attn_sim_crit, attn_sim_trivial)
        #total_loss+=10000*F.mse_loss(attn_critical[:,1, :], attn_critical[:,5, :])
        #total_loss+=10000*F.mse_loss(attn_critical[:,4, :], attn_critical[:,0, :])
        # total_loss +=  F.kl_div(attn_critical[:,1, :].log(), attn_critical[:,5, :], reduction='batchmean')
        # total_loss +=  F.kl_div(attn_critical[:,4, :].log(), attn_critical[:,0, :], reduction='batchmean')
            
        mean_critical_attention = attn_critical.mean(dim=1, keepdim=True)  # Shape: (B, 1, 256)
        # Compute the variance across tokens for each patch
        variance = ((attn_critical - mean_critical_attention) ** 2).mean(dim=1)  # Shape: (B, 256)
        # Average across all patches and batches
        #total_loss += variance.mean()

        # mean_trivial_attention = attn_non_critical.mean(dim=1, keepdim=True)  # Shape: (B, 1, 256)
        # # Compute the variance across tokens for each patch
        # variance = ((attn_non_critical - mean_trivial_attention) ** 2).mean(dim=1)  # Shape: (B, 256)
        # # Average across all patches and batches
        # total_loss += variance.mean()
        
        # total_loss += (mean_trivial_attention*mean_critical_attention).mean()
        return total_loss
    

class SpatialAwarePatchLoss_mine_spatial(nn.Module):
    def __init__(self, gamma=5.0, sigma=1.0):
        """
        Args:
            gamma (float): Controls how strongly distance affects penalties.
            sigma (float): Controls the sharpness of spatial weighting.
        """
        super().__init__()
        self.gamma = gamma
        self.sigma = sigma
        self.lambda_val = 3.0

    def gaussian_kernel(self, H, W, sigma=3.0, device='cuda'):
        """Computes a Gaussian distance weight matrix."""
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        coords = torch.stack([x, y], dim=-1).float().view(-1, 2)  # Shape: (256, 2)

        # Compute pairwise distances
        dist = torch.cdist(coords, coords).pow(2)  # (256, 256)

        # Gaussian weighting
        return torch.exp(-dist / (2 * sigma**2)).unsqueeze(0)  # (1, 256, 256)
    
    
    def compute_distance_matrix(self, H, W, device):
        """Compute normalized spatial distance matrix for a grid (H, W)."""
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1)  # (H, W, 2)
        coords = coords.reshape(-1, 2).to(device)  # (256, 2) for a 16x16 grid

        # Compute pairwise Euclidean distance
        dist_matrix = torch.cdist(coords.float(), coords.float(), p=1)  # (256, 256)

        # Normalize to [0,1] using the largest distance in the grid
        #dist_matrix /= dist_matrix.max()
        return dist_matrix  # (256, 256)


    def forward(self, image_feats, attn_critical, attn_non_critical):
        """
        Args:
            image_feats (torch.Tensor): Image features of hsv image (B, 1024, 16, 16).
            attn_critical (torch.Tensor): attention weights critical (B, 7, 16, 16).

        Returns:
            torch.Tensor: Spatially aware loss.
        """
        B, C, H, W = image_feats.shape
        _, num_critical, _ = attn_critical.shape
        _, num_trivial, _ = attn_non_critical.shape
        num_patches = H * W
        device = image_feats.device

        # Flatten patches
        image_feats = image_feats.view(B, C, num_patches)
        attn_critical = attn_critical.view(B, num_critical, num_patches)
        attn_non_critical = attn_non_critical.view(B, num_trivial, num_patches)

        total_loss = 0
        attn_loss_dict = {}

        ############################################# Fuzzy Similarity with Spatial Awareness
        # Compute pairwise fuzzy similarity
        feat_sim = torch.zeros(B, 256, 256, device=device)
        h1 = image_feats[:, 0, :].unsqueeze(2)  # (B, 256, 1)
        h2 = image_feats[:, 0, :].unsqueeze(1)  # (B, 1, 256)
        diff = torch.abs(h1 - h2)  # (B, 256, 256)
        diff = torch.minimum(diff, 360 - diff)  # Handle hue wrap-around
        hue_sim = torch.exp(-diff ** 2 / (2 * 15 ** 2))  # (B, 256, 256)
        s1, s2 = image_feats[:, 1, :].unsqueeze(2), image_feats[:, 1, :].unsqueeze(1)
        v1, v2 = image_feats[:, 2, :].unsqueeze(2), image_feats[:, 2, :].unsqueeze(1)

        sat_sim = torch.exp(-torch.abs(s1 - s2) / 0.2)  # (B, 256, 256)
        val_sim = torch.exp(-torch.abs(v1 - v2) / 0.2)  # (B, 256, 256)
        feat_sim = 0.7 * hue_sim + 0.2 * sat_sim + 0.1 * val_sim  # (B, 256, 256)

        D = self.compute_distance_matrix(H, W, device).to(device)  # (256, 256)

        # Convert distance into weight (closer patches get higher weights)
        #spatial_weight = torch.exp(- (D ** 2) / (2 * self.sigma ** 2))

        # Apply Laplacian Kernel
        spatial_weight = torch.exp(-D / self.lambda_val).to(device)  # (256, 256)

        # Apply spatial weighting Gaussian
        #spatial_weight = self.gaussian_kernel(H, W, self.sigma, device)  # (1, 256, 256)

        #weighted_feat_sim = feat_sim * spatial_weight  # Reduce long-range false matches
        #weighted_feat_sim = feat_sim * spatial_weight[None, :, :].to(device)  # Shape: (B, 256, 256)  Option to use spatial weights
        #normalized_D = 1 / (1 + D)  # (256, 256), closer patches have higher weight
        #normalized_D = normalized_D / normalized_D.max()  # Normalize to [0,1] for stability

        ############################## Option soft normalization not used right now
        normalized_D = D / (1 + D)  # (256, 256), closer patches have higher weight
        ##############################
        # Define a radius (e.g., within 3 pixels)
        local_radius = 3

        # Mask out distances greater than `local_radius`
        local_mask = (D <= local_radius).float().to(device)  # (256, 256)

        # for i in range(num_critical):
        #     attn_sim_crit = torch.bmm(
        #         attn_critical[:, i, :].unsqueeze(2), attn_critical[:, i, :].unsqueeze(1)
        #     )  # (B, 256, 256)

        #     # Apply local mask
        #     attn_sim_crit_masked = attn_sim_crit * local_mask
        #     feat_sim_masked = feat_sim * local_mask

        #     # Compute loss only in local regions
        #     loss_sim = F.mse_loss(attn_sim_crit_masked, feat_sim_masked, reduction='mean')
        #     total_loss += 10 * loss_sim

        # return total_loss

        #weighted_feat_sim = feat_sim*normalized_D[None, :, :].to(device)
        
        #weighted_feat_sim = feat_sim*D[None, :, :].to(device)
        weighted_feat_sim = feat_sim
        alpha=0.0
        # Compute attention similarity and loss
        for i in range(num_critical):
            # attn_sim_crit = torch.bmm(attn_critical[:, i, :].unsqueeze(2), attn_critical[:, i, :].unsqueeze(1))  # (B, 256, 256)
            # loss_sim = F.mse_loss(attn_sim_crit, weighted_feat_sim)
            # total_loss += 10 * loss_sim
            # continue
            # if i != 0:
            #     continue
            if i < 10:
                attn_sim_crit = torch.bmm(attn_critical[:, i, :].unsqueeze(2), attn_critical[:, i, :].unsqueeze(1))  # (B, 256, 256)
                #loss_sim = F.mse_loss(attn_sim_crit, feat_sim)
                loss_sim = F.mse_loss(attn_sim_crit, weighted_feat_sim, reduction='none')
                #loss_sim = alpha*F.mse_loss(attn_sim_crit, weighted_feat_sim) + (1-alpha)*F.mse_loss(attn_sim_crit, feat_sim)
                total_loss += 10 * loss_sim
            else:
                total_loss += 10* F.mse_loss(attn_critical[:, 0, :], attn_critical[:, i, :])

        # Encourage smoothness in attention maps (low variance)
        # mean_critical_attention = attn_critical.mean(dim=1, keepdim=True)  # (B, 1, 256)
        # variance = ((attn_critical - mean_critical_attention) ** 2).mean(dim=1)  # (B, 256)
        # total_loss += variance.mean()
        attn_loss_dict["color_loss"] = total_loss

        return attn_loss_dict

def retinal_loss_function(attention_map, crit_dot, triv_dot, hsv_patches, lambda1=0.5, lambda2=1.0, lambda3=0.1):
    """
    Computes the loss enforcing:
    1) Critical and trivial tokens should have minimal overlap in their dot products with feature embeddings.
    2) Certain critical tokens should activate on specific color-based patches.
    3) Trivial tokens should be automatically reduced if unnecessary.
    
    Parameters:
        T_crit: Tensor of shape (7, 1024) - Critical tokens
        T_triv: Tensor of shape (n_t, 1024) - Trivial tokens
        F_vit: Tensor of shape (256, 1024) - Vision Transformer feature embeddings
        hsv_patches: Tensor of shape (256, 3) - HSV values of each patch
        lambda1, lambda2, lambda3: Weighting factors for each loss term
    """
    hue = hsv_patches[:, 0, :]  # Shape (B, 256)
    saturation = hsv_patches[:, 1, :]  # Shape (B, 256)
    value = hsv_patches[:, 2, :]  # Shape (B, 256)

    # Compute dot products (before softmax)
    # crit_dot = torch.matmul(T_crit, F_vit.T)  # (7, 256)
    # triv_dot = torch.matmul(T_triv, F_vit.T)  # (n_t, 256)
    # Extract the saturation (S) and value (V) channels (channels 1 and 2 in HSV format)
    doing_loss_against_crit_on_black = True

    attn_loss_dict = {}
    if doing_loss_against_crit_on_black:
        threshold=1e-3
        
        # Identify black patches (both S and V are below the threshold)
        black_patch_mask = (hue < -1.68) & (saturation < -1.68) & (value < -1.4)  # Shape (B, 256)
        
        # Now, we want to penalize high attention values where the patch is black
        # Expand black_patch_mask to shape (B, N, 256) to match the attention map's shape
        black_patch_mask_expanded = black_patch_mask.unsqueeze(1).expand(-1, attention_map.shape[1], -1)  # Shape (B, N, 256)
        
        # Apply the mask to the attention map and calculate the loss
        # The loss will be the sum of attention values for black patches
        masked_attention = attention_map * black_patch_mask_expanded.float() # Apply mask
        
        # Loss is the sum of masked attention values
        black_crit_loss = masked_attention.sum()  # Sum over all patches and batches
        attn_loss_dict["black_crit_loss"] = black_crit_loss

    
    # 1) Critical-Trivial Overlap Loss (Minimize dot product similarity)
    L_overlap = torch.sum(torch.matmul(crit_dot, triv_dot.permute(0, 2, 1))) / (crit_dot.shape[0] * crit_dot.shape[1] * triv_dot.shape[1])  # Normalize by token count
    attn_loss_dict["overlap_loss"] = L_overlap

    return attn_loss_dict
    if L_overlap<-800:
        L_overlap*=0
    # 2) Color-Based Activation Loss
    # H, S, V = hsv_patches[:, 0, :], hsv_patches[:, 1, :], hsv_patches[:, 2, :]  # Extract HSV channels
    
    # Define binary masks based on HSV conditions (1 = should activate, 0 = should not)
    M_exudates = ((hue >= 30) & (hue <= 60)).float()  # Yellow regions
    M_hemorrhages = (((hue <= 10) | (hue >= 350)) & (value < 0.5)).float()  # Dark red regions
    M_microaneurysms = ((hue <= 10) & (saturation > 0.6)).float()
    M_neovascularization = torch.ones_like(hue)  # Placeholder (should use SSIM-based mask)
    M_edema = ((saturation < 0.3) & (value < 0.5)).float()  # Dark & low saturation
    
    # Stack masks in expected critical token order
    M = torch.stack([M_microaneurysms, M_hemorrhages, M_exudates, M_neovascularization, M_edema], dim=1)
    #print(M.shape)
    #print(crit_dot.shape)
    
    normalized_attention = attention_map / attention_map.max(dim=1, keepdim=True).values
    L_color = F.mse_loss(normalized_attention[:,:normalized_attention.shape[1],:], M, reduction="none")  # Compare first 5 critical tokens to their expected maps
    # Enforce that critical tokens match expected color patches
    #L_color = F.mse_loss(crit_dot[:,:crit_dot.shape[1],:], M, reduction="none")  # Compare first 5 critical tokens to their expected maps
    #print(f"L color shape {L_color.shape}")
    attn_loss_dict["color_loss"] = L_color
    #L_color = 1 - F.cosine_similarity(crit_dot[:,:crit_dot.shape[1],:], M, dim=-1).mean()
    
    # 3) Automatic Trivial Token Reduction (Minimize their dot products)
    L_sparsity = torch.mean(torch.abs(triv_dot))
    attn_loss_dict["sparsity_loss"] = L_sparsity
    
    # Final loss
   # loss = lambda1 * L_overlap + lambda2 * L_color + lambda3 * L_sparsity
    
    return attn_loss_dict

def attention_variance_loss(attention_maps, window_size=3):
    # attention_maps: Tensor of shape (B, N_c, T), where B is the batch size, 
    # N_c is the number of tokens, and T is the number of patches.
    
     # attention_maps: Tensor of shape (B, N_c, T) - original attention maps
    B, N_c, T = attention_maps.shape
    
    # Assuming T is a square (H * W), so reshaping to (B, N_c, H, W)
    H = W = int(T**0.5)  # Assuming T is a perfect square
    attention_maps = attention_maps.view(B, N_c, H, W)
    
    # Apply a 2D sliding window using unfold
    unfold = F.unfold(attention_maps, kernel_size=window_size, stride=1, padding=window_size//2)
    
    # Unfolding returns (B, N_c * window_size * window_size, H * W), we need to calculate variance for each window
    unfold = unfold.view(B, N_c, window_size*window_size, H, W)
    
    # Calculate mean and variance within each window
    mean_unfold = unfold.mean(dim=2, keepdim=True)  # Mean across the window_size * window_size dimension
    var_unfold = ((unfold - mean_unfold) ** 2).mean(dim=2)  # Variance across the window dimension
    
    # Average variance across all windows
    loss = var_unfold.mean()
    
    return loss

def skin_lesion_loss_function(attention_map, crit_dot, triv_dot, hsv_patches, lambda1=0.5, lambda2=1.0, lambda3=0.1):
    # 2) Color-Based Activation Loss (Different masks for different labels)
    H = hsv_patches[:, 0, :]  # Shape (B, 256)
    S = hsv_patches[:, 1, :]  # Shape (B, 256)
    V = hsv_patches[:, 2, :]  # Shape (B, 256)

    # Define masks for different lesion types in batch form
    masks = torch.stack([
        ((H >= 0) & (H <= 30)).float(),  # Actinic Keratosis (Red/pink/brown)
        ((H >= 200) & (H <= 250)).float(),  # Basal Cell Carcinoma (Translucent, pearly white)
        ((H >= 0) & (H <= 30)).float(),  # Benign Keratosis-like (Red/pink/brown)
        ((H >= 30) & (H <= 100)).float(),  # Dermatofibroma (Light brown to black)
        ((H >= 0) & (H <= 360)).float(),  # Melanoma (Variable colors)
        ((H >= 20) & (H <= 50)).float(),  # Nevus (Uniformly tan/brown/black)
        ((H >= 300) & (H <= 360)).float(),  # Vascular Lesion (Red, purple, blue)
    ], dim=1)  # (B, num_classes, 256)

    # 3) **Critical Token Similarity Loss**
    L_sim = torch.mean(torch.abs(crit_dot[:,[1,2,4,5,6],:] - crit_dot[:,0,:].detach().unsqueeze(1)))  # L1 loss for similarity
    return L_sim, masks, crit_dot  # Return mask losses separately if you need to select them later

def attention_map_zero_and_three_loss(attention_maps):
    """
    attention_maps: Tensor of shape (B, N_c, T)
    - B: batch size
    - N_c: number of tokens (number of maps)
    - T: number of patches
    
    This function computes a loss that detaches the attention maps at indices 0 and 3,
    and encourages the other attention maps to be similar to them using a similarity loss.
    """
    # Get the maps at indices 0 and 3 (detached)
    detached_maps = attention_maps[:, [0, 3], :]  # Shape: (B, 2, T)
    
    # Detach the attention maps at indices 0 and 3 (no gradient)
    detached_maps = detached_maps.detach()

    # Get the other maps (exclude indices 0 and 3)
    remaining_maps = attention_maps[:, [i for i in range(attention_maps.shape[1]) if i not in [0, 3]], :]  # Shape: (B, N_c-2, T)
    
    # Calculate the similarity loss (e.g., MSE loss)
    # We want to make the remaining maps similar to the detached maps at indices 0 and 3
    
    # For simplicity, we will use the mean of the maps at indices 0 and 3
    target_map = detached_maps.mean(dim=1)  # Shape: (B, T) -> Taking the mean of maps at indices 0 and 3
    
    # Compute the MSE loss between the remaining maps and the target map
    mse_loss = F.mse_loss(remaining_maps, target_map.unsqueeze(1).expand(-1, remaining_maps.shape[1], -1))  # Expand to match dimensions
    
    return mse_loss

def smoothness_loss(attention_maps):
    """
    Computes a smoothness loss for attention maps of shape (B, N_c, T).
    
    Args:
        attention_maps: Tensor of shape (B, N_c, T).
        T: Number of patches (must be a perfect square).
        
    Returns:
        Smoothness loss: Tensor.
    """
    B, N_c, T = attention_maps.shape
    
    # Assume T is a perfect square and reshape the patches into a 2D grid.
    side_length = int(T**0.5)
    attention_maps_reshaped = attention_maps.view(B, N_c, side_length, side_length)
    
    # Compute differences between adjacent patches (both horizontally and vertically).
    diff_horiz = attention_maps_reshaped[:, :, :, 1:] - attention_maps_reshaped[:, :, :, :-1]
    diff_vert = attention_maps_reshaped[:, :, 1:, :] - attention_maps_reshaped[:, :, :-1, :]
    
    # Compute the smoothness loss (mean squared difference for horizontal and vertical differences).
    loss_horiz = torch.mean(diff_horiz ** 2)
    loss_vert = torch.mean(diff_vert ** 2)
    
    # Total smoothness loss is the sum of the horizontal and vertical components.
    total_loss = loss_horiz + loss_vert
    
    return total_loss


def total_non_overlap_loss(attn_c, attn_t, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    Compute the total loss enforcing non-overlapping attention maps.
    
    Args:
        attn_c: Tensor of shape (B, N_c, T) - Critical token attention maps.
        attn_t: Tensor of shape (B, N_t, T) - Trivial token attention maps.
        lambda1, lambda2, lambda3: Weights for the different losses.
    
    Returns:
        Scalar loss value.
    """
    # Compute dot product between attention maps across the patch dimension
    overlap =torch.einsum('bnt,bmt->bt', attn_c, attn_t)  # Shape: (B, T)
    overlap_loss = overlap.mean()

    total_attention = attn_c.sum(dim=1) + attn_t.sum(dim=1)  # Sum over N_c and N_t, shape: (B, T)
    sum_penalty_loss = (total_attention ** 2).mean()  # Minimize squared sum

    # Normalize to get probability distributions over patches
    eps=1e-8
    p_c = attn_c.sum(dim=1, keepdim=True) / (attn_c.sum(dim=(1, 2), keepdim=True) + eps)  # Shape: (B, 1, T)
    p_t = attn_t.sum(dim=1, keepdim=True) / (attn_t.sum(dim=(1, 2), keepdim=True) + eps)  # Shape: (B, 1, T)
    kl_divergence_loss = F.kl_div(p_c.log(), p_t, reduction='batchmean') + F.kl_div(p_t.log(), p_c, reduction='batchmean')

    loss = (
        lambda1 * overlap_loss +
        lambda2 * sum_penalty_loss +
        lambda3 * kl_divergence_loss
    )
    return loss

def image_to_patches(images, patch_size=14):
    """
    Splits an image (3, 224, 224) into patches of size (256, D), where D = 14*14*3.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (3, 224, 224).
        patch_size (int): Size of each patch along one dimension.

    Returns:
        torch.Tensor: Tensor of shape (256, D) where D = patch_size * patch_size * 3.
    """
    # Ensure image is a tensor with shape (B, 3, 224, 224)
    B, C, H, W = images.shape
    num_patches = (H // patch_size) ** 2
    # Unfold each image into patches (B, 3, 16, 16, 14, 14)
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    # Rearrange dimensions to (B, 16, 16, 3, 14, 14) -> (B, 256, 3, 14, 14)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_patches, C, patch_size, patch_size)

    # Flatten each patch into a feature vector (B, 256, D) where D = 14*14*3
    patches = patches.view(B, num_patches, -1)  # Shape: (B, 256, 588)
    #print(patches.shape)
    
    return patches

def rgb_to_hsv_torch(rgb_tensor):
    """
    Convert a PyTorch tensor from RGB to HSV.
    
    Args:
        rgb_tensor (torch.Tensor): Tensor of shape (B, 3, H, W) with values in [0, 1].
    
    Returns:
        torch.Tensor: HSV tensor of shape (B, 3, H, W) with H in [0, 1], S in [0, 1], V in [0, 1].
    """
    r, g, b = rgb_tensor[:, 0, :, :], rgb_tensor[:, 1, :, :], rgb_tensor[:, 2, :, :]

    # Compute Value (Brightness)
    v = torch.max(rgb_tensor, dim=1)[0]

    # Compute Saturation
    min_rgb = torch.min(rgb_tensor, dim=1)[0]
    s = (v - min_rgb) / (v + 1e-10)  # Avoid division by zero

    # Compute Hue
    delta = v - min_rgb + 1e-10  # Small epsilon to prevent NaN
    h = torch.zeros_like(v)

    # Conditions for hue computation
    mask_r = (v == r)
    mask_g = (v == g)
    mask_b = (v == b)

    h[mask_r] = (g[mask_r] - b[mask_r]) / delta[mask_r] % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4

    h = h / 6  # Normalize hue to [0,1]

    # Stack H, S, V back together
    hsv_tensor = torch.stack([h, s, v], dim=1)

    #print(f"hsv tensor {hsv_tensor[0,:,:,:]}")
    
    return hsv_tensor


def attn_chat_gpt_crit_smooth_loss(attn_map, merge_patches=True, merge_factor=2):
    """
    attn_map: (B, N_tokens, N_patches) e.g., (B, 7, 256)
    Assumes square grid: 256 -> 16x16
    """
    # attention_maps: Tensor of shape (B, N_c, T) - original attention maps
    B, N_c, T = attn_map.shape
    
    # Assuming T is a square (H * W), so reshaping to (B, N_c, H, W)
    H = W = int(T**0.5)  # Assuming T is a perfect square
    #attention_maps = attn_map.view(B, N_c, H, W)

    if merge_patches:
        # Downsample using average pooling (2x2 blocks → 1)
        attn_grid_merged = F.avg_pool2d(attn_map.view(B, N_c, H, W), kernel_size=merge_factor)
        # Flatten back
        attn_map = attn_grid_merged.flatten(2)  # (B, N, new_T)
        _, _, T = attn_map.shape
        H = W = int(T**0.5)  # Assuming T is a perfect square

    attn_map_tot_var = attn_map.view(B, N_c, H, W)  # (B, 7, 16, 16)

    tv_loss = (
        torch.mean(torch.abs(attn_map_tot_var[:, :, :, :-1] - attn_map_tot_var[:, :, :, 1:])) +
        torch.mean(torch.abs(attn_map_tot_var[:, :, :-1, :] - attn_map_tot_var[:, :, 1:, :]))
    )*10e4

    #print(f"tv loss {tv_loss}")

    laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]]]], dtype=torch.float32).to(attn_map.device)
    
    attn_map_b_lap = attn_map.view(B * N_c, 1, H, W)  # merge batch and tokens
    laplacian = F.conv2d(attn_map_b_lap, laplacian_kernel, padding=1)
    laplacian_smoothness_loss = torch.mean(torch.abs(laplacian))*10e3*0.5
    #print(f"laplacian smoothness loss {laplacian_smoothness_loss}")

    eps = 1e-8
    entropy = - (attn_map * (attn_map + eps).log()).sum(dim=-1)  # (B, T)
    #print(f"entropy {entropy.mean()*0.1}")

    return tv_loss, laplacian_smoothness_loss, entropy.mean()*0.1

def attn_chat_gpt_crit_different_maps(critical_attn, trivial_attn, merge_patches=True, merge_factor=2):
    """
    critical_attn, trivial_attn: (B, T, P) where T = 7, P = 256
    """
    B, N_c, T = critical_attn.shape
    _, N_t, _ = trivial_attn.shape
    temperature=0.1

    if merge_patches:
        H = W = int(T**0.5)  # Assuming T is a perfect square
        # Downsample using average pooling (2x2 blocks → 1)
        critical_attn_merged = F.avg_pool2d(critical_attn.view(B, N_c, H, W), kernel_size=merge_factor)
        # Flatten back
        critical_attn = critical_attn_merged.flatten(2)  # (B, N, new_T)
        _, _, T = critical_attn.shape

        trivial_attn_merged = F.avg_pool2d(trivial_attn.view(B, N_t, H, W), kernel_size=merge_factor)
        # Flatten back
        trivial_attn = trivial_attn_merged.flatten(2)  # (B, N, new_T)

    # Flatten across batch
    crit_emb = critical_attn.view(B * N_c, T)  # (B*T_crit, P)
    triv_emb = trivial_attn.view(B * N_t, T)   # (B*T_triv, P)

    # Normalize embeddings
    crit_emb = F.normalize(crit_emb, dim=-1)
    triv_emb = F.normalize(triv_emb, dim=-1)

    # Compute similarity matrix (cosine similarity)
    sim_matrix = torch.mm(crit_emb, triv_emb.T)  # (B*T_crit, B*T_triv)

    # Contrastive loss (NT-Xent)
    logits = sim_matrix / temperature  # Scale similarity
    labels = torch.arange(len(crit_emb)).to(crit_emb.device) % len(triv_emb)  # Matching pairs

    contrast_loss = F.cross_entropy(logits, labels)  # High similarity → lower loss
    ##############################
    #crit_flat = critical_attn.view(B, -1)
    #triv_flat = trivial_attn.view(B, -1)
    #dot_product = (crit_flat * triv_flat).sum(-1)  # Sum across patches
    #orthog_loss = dot_product.mean()*10e2  # Encourage dot product ≈ 0
    pairwise_mul = critical_attn.unsqueeze(2) * trivial_attn.unsqueeze(1)  # (B, N_c, N_t, T)
    orthog_loss = pairwise_mul.sum(-1).mean()*10e2  # (B, N_c, N_t) → scalar
    ###############################
    crit = F.softmax(critical_attn, dim=-1)
    triv = F.softmax(trivial_attn, dim=-1)
    kl1 = F.kl_div(triv.log(), crit, reduction='batchmean')
    kl2 = F.kl_div(crit.log(), triv, reduction='batchmean')
    kl_loss = -(kl1 + kl2)*0  # Encourage separation
    ###############################
    crit = F.softmax(critical_attn, dim=-1)
    triv = F.softmax(trivial_attn, dim=-1)
    inv_loss = (crit * triv).sum(-1).mean()*10e3  # Encourage dissimilarity

    #print(f"contrast loss {contrast_loss}")

    crit = F.normalize(crit, dim=-1)
    triv = F.normalize(triv, dim=-1)
    dot_loss = (crit.unsqueeze(2) * triv.unsqueeze(1)).sum(-1).mean()
    #print(f"dot loss {dot_loss}")

    # crit: (B*N_c, T), triv: (B*N_t, T)
    dist = 1 - F.cosine_similarity(crit, triv)
    dist_loss = torch.clamp(0.5 - dist, min=0).mean()
    #print(f"dist loss {dist_loss}")


    #print(f"orthog loss {orthog_loss}")
    #print(f"kl loss {kl_loss}")
    #print(f"inv loss {inv_loss}")
    return contrast_loss, orthog_loss, kl_loss, inv_loss, dot_loss, dist_loss

def vectorized_similarity_loss(attn_maps):
    """
    attn_maps: Tensor of shape (B, N_c, T), assumed L2 normalized
    """
    attn_maps = F.normalize(attn_maps, p=2, dim=-1)
    _, N_c, _ = attn_maps.shape

    # (B, N_c, T) @ (B, T, N_c) → (B, N_c, N_c) pairwise similarities
    sim_matrix = torch.bmm(attn_maps, attn_maps.transpose(1, 2))  # cosine sim

    # Remove self-similarity (diagonal)
    mask = torch.eye(N_c, device=attn_maps.device).bool()
    sim_matrix.masked_fill_(mask[None, :, :], 0.0)

    # Average pairwise 1 - sim across upper triangle
    num_pairs = N_c * (N_c - 1)
    sim_loss = (1 - sim_matrix).sum(dim=(1, 2)) / num_pairs  # (B,)
    return sim_loss.mean()

def anchor_alignment_loss(attn_maps):
    """
    attn_maps: (B, N_c, T), L2 normalized
    Token 0 is anchor — others align to it, but it's frozen.
    """
    anchor = attn_maps[:, 0:1, :].detach()  # (B, 1, T)
    others = attn_maps[:, 1:, :]            # (B, N_c-1, T)

    # Use valid einsum notation: 'bat, bct -> bc' where a=1 (anchor), c=N_c-1
    sim = torch.einsum('bat,bct->bc', anchor, others)  # (B, N_c-1)
    loss = (1 - sim).mean()
    return loss

def mutual_exclusivity_loss(attn_maps):
    """
    attn_maps: (B, C, N) - attention values for C concepts on N patches

    Returns: scalar loss that discourages multiple concepts attending to same patch
    """
    B, C, N = attn_maps.shape

    # Normalize attention maps over concepts (so each patch's attention across concepts sums to 1)
    attn_soft = F.softmax(attn_maps, dim=1)  # (B, C, N)

    # For each patch, we want softmax over C to be close to one-hot
    # Use entropy over concepts at each patch
    entropy = -torch.sum(attn_soft * torch.log(attn_soft + 1e-8), dim=1)  # (B, N)

    # Max entropy would be log(C), so we normalize
    entropy = entropy / torch.log(torch.tensor(C, dtype= attn_maps.dtype, device=attn_maps.device))

    return entropy.mean(dim=1)  # (B,)

def high_attention_penalty(attn_map, max_tokens=2):
    """
    Loss function to enforce that at most `max_tokens` tokens have high attention in each patch.
    
    Parameters:
    - attn_map: Tensor of shape (B, N_c, T) representing attention maps.
    - max_tokens: Maximum number of tokens that should have high attention in each patch.
    
    Returns:
    - loss: A scalar value representing the penalty for violating the condition.
    """
    B, N_c, T = attn_map.shape
    
    # Reshape the attention map to (B * T, N_c) so we can work with each patch (in terms of its tokens)
    attn_map_flattened = attn_map.view(B * T, N_c)
    
    # For each patch, find the top-k tokens with the highest attention
    top_k_values, top_k_indices = torch.topk(attn_map_flattened, max_tokens, dim=-1, largest=True)
    
    # Compute the total number of tokens in each patch that exceed the max_tokens threshold
    # Count how many values in each patch exceed the top-k values
    num_high_tokens = torch.sum(attn_map_flattened > top_k_values[:, -1].unsqueeze(1), dim=-1)
    
    # Penalize if there are more than `max_tokens` high tokens in a patch
    penalty = torch.sum(torch.relu(num_high_tokens - max_tokens))
    
    # Return the total penalty as the loss
    return penalty

def mse_loss_between_maps(attn_map):
    """
    Compute the MSE loss between all pairs of attention maps in the batch without using for loops.
    
    Parameters:
    - attn_map: Tensor of shape (B, N_c, T) representing the attention maps.
    
    Returns:
    - loss: Scalar value representing the MSE loss between attention maps in the batch.
    """
    B, N_c, T = attn_map.shape
    
    # Reshape the attention map to (B, N_c * T) for easy pairwise comparison
    attn_map_flattened = attn_map.view(B, -1)  # Shape: (B, N_c * T)
    
    # Compute pairwise MSE loss using broadcasting
    # Calculate the squared difference between each pair of attention maps
    diff = attn_map_flattened.unsqueeze(0) - attn_map_flattened.unsqueeze(1)  # Shape: (B, B, N_c * T)
    
    # Calculate the squared error
    squared_diff = diff ** 2
    
    # Compute the mean squared error over each pair of maps
    mse_loss = squared_diff.mean(dim=-1)  # Shape: (B, B) -> MSE between each pair
    
    # Exclude the diagonal (self comparisons) by masking
    mask = torch.eye(B, device=attn_map.device).bool()
    mse_loss = mse_loss[~mask].mean()  # Flatten the loss and take the average
    
    return mse_loss

class ExpLICD_ViT_L_Classic_with_Spatial_Bias(nn.Module):  
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.num_of_criteria = NUM_OF_CRITERIA[config.dataset]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.new_explicd = config.new_explicd
        self.trivial_ratio = config.trivial_ratio
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
            
            config.preprocess = preprocess

            self.model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual

            #################
            old_pos_emb = self.model_visual_ViT_L.positional_embedding  # (257, 1024) including CLS token
            cls_emb, grid_emb = old_pos_emb[:1], old_pos_emb[1:]  # Split CLS token

            # Reshape the grid positional embeddings into (16, 16, embedding_dim)
            embedding_dim = grid_emb.shape[1]
            grid_emb = grid_emb.reshape(16, 16, embedding_dim).permute(2, 0, 1)  # (1024, 16, 16)

            # Interpolate to (32, 32)
            new_size = (32, 32)
            new_grid_emb = torch.nn.functional.interpolate(grid_emb.unsqueeze(0), size=new_size, mode="bilinear", align_corners=False).squeeze(0)

            # Reshape back to (1024, 32×32)
            new_grid_emb = new_grid_emb.permute(1, 2, 0).reshape(-1, embedding_dim)

            # Concatenate CLS token back
            new_pos_emb = torch.cat([cls_emb, new_grid_emb], dim=0)  # (1025, 1024)

            # Update model's positional embeddings
            self.model_visual_ViT_L.positional_embedding = torch.nn.Parameter(new_pos_emb)
            #################
            
            ######################### Option use additional ViT
            if self.new_explicd==1:
                self.model_visual_ViT_L_hsv = clip.load(f"ViT-L/14")[0].visual
                for param in self.model_visual_ViT_L_hsv.parameters():
                    param.data = param.data.to(torch.float32)
                self.model_visual_ViT_L_hsv.cuda()
            #########################
            #self.model_visual_custom = VisionTransformer7x7()

            #Convert specific layers or parameters to float32
            for param in self.model_visual_ViT_L.parameters():
                param.data = param.data.to(torch.float32)

            self.model_visual_ViT_L.cuda()
            #self.model_visual_custom.cuda()
            
            self.model.cuda()
            
            concept_keys = list(concept_list.keys())
            self.layers = nn.ModuleDict({
            })
            self.concept_token_dict = {}
            for key in concept_keys:
                prefix = get_prefix(self.config.dataset, key)
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()
                self.layers[key] = nn.Sequential(te.nn.EntropyLinear(len(attr_concept_list), len(attr_concept_list)*3, n_classes=len(attr_concept_list)),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(len(attr_concept_list)*3, len(attr_concept_list)*1),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(len(attr_concept_list)*1, 1)
                            )

            
            self.logit_scale = logit_scale.detach()

            ################ Option Clip L
            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
            ################
        
        self.visual_features = []
        if self.new_explicd==1:
            self.visual_features_hsv = []
        self.hook_list = []
        if self.new_explicd==1:
            self.hook_list_hsv = []

        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
            
        def hook_fn_hsv(module, input, output):
            self.visual_features_hsv.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        
        ################# Option BioMedclip
        # layers = [self.model.visual.trunk.blocks[11]]

        # for param in self.model.text.parameters():
        #     param.requires_grad = False
        # for param in self.model.visual.trunk.parameters():
        #     param.requires_grad = True
        ################# Option Clip L
        layers = [self.model_visual_ViT_L.transformer.resblocks[23]]
        if self.new_explicd==1:
            layers_hsv = [self.model_visual_ViT_L_hsv.transformer.resblocks[23]]
        #################

        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))

        if self.new_explicd==1:
            for layer in layers_hsv:
                self.hook_list_hsv.append(layer.register_forward_hook(hook_fn_hsv))

        hidden_size = 1024
        num_heads = 16
        self.critical_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, hidden_size, dtype=torch.float32)))
        self.trivial_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(int(self.num_of_criteria*self.trivial_ratio), hidden_size, dtype=torch.float32)))

        self.critical_visual_tokens_smaller = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, hidden_size, dtype=torch.float32)))
        self.trivial_visual_tokens_smaller  = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, hidden_size, dtype=torch.float32)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn_critical = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attns_criticals = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True) for _ in range(self.num_of_criteria)])
        self.cross_attn_trivial = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
    
        self.self_attn_critical = Attention(hidden_size, num_heads=num_heads, qkv_bias=False, qk_norm=True)

        self.cross_attn_critical_smaller = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn_trivial_smaller = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

        self.cross_attn_between_patches = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn_in_patches = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ffn = FFN(hidden_size, hidden_size*4)

        self.ffn_smaller = FFN(hidden_size, hidden_size*4)
        self.norm_smaller = nn.LayerNorm(hidden_size)

        self.norm_before = nn.LayerNorm(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(in_features=hidden_size, out_features=512, bias=False)

        dim_of_sit_tokens = 768
        num_heads_sit = 16
        self.proj_tokens_for_Sit = nn.Linear(in_features=hidden_size, out_features=dim_of_sit_tokens, bias=False)
        self.ffn_vit_output = FFN(hidden_size, hidden_size*4)
        self.proj_vit_output = nn.Linear(in_features=hidden_size, out_features=dim_of_sit_tokens, bias=False)
        self.norm_vit_output = nn.LayerNorm(hidden_size)
        self.norm_tokens_for_Sit = nn.LayerNorm(dim_of_sit_tokens)
        self.cross_attn_critical_for_Sit = nn.MultiheadAttention(embed_dim=dim_of_sit_tokens, num_heads=num_heads_sit, batch_first=True)

        self.proj_Sit = nn.Linear(in_features=hidden_size, out_features=768, bias=False)
        #self.proj_Sit_to_explicid = nn.Linear(in_features=768, out_features=512, bias=False)

        #                                     34  for ISIC
        #                                     17 for BUSI
        self.cls_head = nn.Linear(in_features=NUM_OF_SIMILARITIES[self.config.dataset], out_features=config.num_class)
        self.minimal_cls_head = nn.Linear(in_features=hidden_size*7, out_features=config.num_class)
        self.cls_wiht_te =  nn.Sequential(te.nn.EntropyLinear(NUM_OF_SIMILARITIES[self.config.dataset], NUM_OF_SIMILARITIES[self.config.dataset]*5, n_classes=config.num_class),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(NUM_OF_SIMILARITIES[self.config.dataset]*5, config.num_class*4),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(config.num_class*4, 1)
                            )

        self.cluster_sigma = torch.nn.Parameter(torch.ones(1))
        self.orthogonal_sigma = torch.nn.Parameter(torch.ones(1))
        self.coverage_sigma = torch.nn.Parameter(torch.ones(1))

        for param in self.model_visual_ViT_L.parameters():
            param.requires_grad = True

        if self.new_explicd==1:
            for param in self.model_visual_ViT_L_hsv.parameters():
                param.requires_grad = True

        self.critical_visual_tokens.requires_grad = True
        self.trivial_visual_tokens.requires_grad = True

        self.critical_visual_tokens_smaller.requires_grad = True
        self.trivial_visual_tokens_smaller.requires_grad = True

        ###########################################################
        # self.cnn = PatchSelectorCNN().cuda()
        # self.cnn_kernel_14 = PatchSelectorCNN_kernel_14().cuda()
        # self.classifying_critical_cnn = ClassifyingCNN().cuda()
        #self.classifying_trivial_cnn = ClassifyingCNN().cuda()
        #self.gnn =  CircularConv(in_dim=1024, out_dim=1024).cuda()
        #self.node_classifier =  NodeClassifier(in_dim=1024).cuda()

    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.critical_visual_tokens)
        param_list.append(self.trivial_visual_tokens)
        for param in self.cross_attn.parameters():
            param_list.append(param)

        for param in self.ffn.parameters():
            param_list.append(param)
        for param in self.norm.parameters():
            param_list.append(param)
        for param in self.proj.parameters():
            param_list.append(param)

        for i in range(self.num_of_criteria):
            for param in self.ffns[i].parameters():
                param_list.append(param)
            for param in self.norms[i].parameters():
                param_list.append(param)
            for param in self.projs[i].parameters():
                param_list.append(param)
            for param in self.proj_Sits[i].parameters():
                param_list.append(param)
            for param in self.proj_Sits_to_explicids[i].parameters():
                param_list.append(param)

        for param in self.cls_head.parameters():
            param_list.append(param)

        return param_list


    def forward(self, imgs, refined_tokens=None, explicid_imgs_latents=None, imgs_wo_norm=None):
        
        if refined_tokens is not None:
            image_logits_dict_refined = {}
            idx = 0
            refined_tokens = self.proj_Sit_to_explicid(F.normalize(refined_tokens, dim=-1))
            for key in self.concept_token_dict.keys():
                image_logits_dict_refined[key] = (self.logit_scale * refined_tokens[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(refined_tokens.shape[0], 1, 1).permute(0, 2, 1)).squeeze(1)
                idx += 1
            image_logits_list_refined = []
            for key in image_logits_dict_refined.keys():
                image_logits_list_refined.append(image_logits_dict_refined[key])
        
            image_logits_refined = torch.cat(image_logits_list_refined, dim=-1)
            cls_logits_refined = self.cls_head(image_logits_refined)
            return cls_logits_refined

        ###################################### Option to use vit twice
        if self.new_explicd==1:
            self.visual_features_hsv.clear()
        
        # if imgs_wo_norm is not None:
        #     imgs_in_hsv = rgb_to_hsv_torch(imgs_wo_norm)
        # else:
        #     imgs_in_hsv = imgs
        imgs_in_hsv = imgs
        img_patches = imgs_in_hsv.unfold(2, 14, 14).unfold(3, 14, 14)  # (B, 3, 16, 16, 14, 14)
        img_patches = img_patches.mean(dim=(-1, -2))  # (B, 3, 16, 16)
        if self.new_explicd==1:
            _ = self.model_visual_ViT_L_hsv(imgs_in_hsv)
        patches = image_to_patches(imgs, patch_size=14)
        #print(f"shape of self.visual_features_hsv[0] {self.visual_features_hsv[0].shape}")
        #vit_l_output_hsv=self.visual_features_hsv[0][:, 1:, :]
        if self.new_explicd==1:
            vit_l_output_hsv=self.visual_features_hsv[0].permute(1, 0, 2)[:, 1:, :]
        #vit_l_output_hsv = 
        ######################################


        #patches = image_to_patches(imgs_in_hsv, patch_size=14)
        B, T, D_patch = patches.shape  # B: batch size, T: num_patches (256), D: patch_dim (1024)
        #print(patches.shape)
        ################# Option BioMedclip
        # _, _, _ = self.model(imgs, None)
        # vit_l_output = self.visual_features[0][:, 1:, :]
        ################# Option Clip L
        ################# Option gradual
        self.visual_features.clear()
        _ = self.model_visual_ViT_L(imgs)
        #critical_mask, trivial_mask = self.cnn_kernel_14(imgs)  # Shape: (B, 1, 16, 16)
        #loss_fn = SpatialAwarePatchLoss()
        attn_explicd_loss = torch.tensor(0.0, device=self.device)
        #attn_explicd_loss += 10*loss_fn(patches.permute(0, 2, 1).view(B, D_patch, int(T ** 0.5), int(T ** 0.5)), critical_mask)
        
        # vit_l_output = self.model_visual_ViT_L.conv1(imgs)  # Convolutional layer
        # vit_l_output = vit_l_output.flatten(2).transpose(1, 2)  # Flatten and transpose
        # vit_l_output = self.model_visual_ViT_L.ln_pre(vit_l_output)  # Layer normalization
        # vit_l_output = vit_l_output + self.model_visual_ViT_L.positional_embedding[:vit_l_output.size(1), :]  # Positional embedding
        # for i in range(23):
        #     vit_l_output = self.model_visual_ViT_L.transformer.resblocks[i](vit_l_output)  # First 7 transformer blocks
        #print(vit_l_output.shape)
        ################# Option instanentous
        #print(f"shape of self.visual_features[0] {self.visual_features[0].shape}")
        vit_l_output=self.visual_features[0].permute(1, 0, 2)[:, 1:, :]
        if self.new_explicd==0:
            vit_l_output_hsv = vit_l_output
        #################
        B, T, D = vit_l_output.shape  # B: batch size, T: num_patches (256), D: patch_dim (1024)
        H = W = int(T ** 0.5)  # Assuming T is a perfect square, H = W = 16
        ############################################################################   Option use Attention Loss for the tokens
        vit_output_unflattened = vit_l_output.view(B, H, W, D).permute(0, 3, 1, 2)  # Shape: (B, 1024, 16, 16)
        #vit_output_unflattened_for_exlusive = vit_output_unflattened.clone().detach()
        #vit_output_unflattened_for_exlusive =vit_output_unflattened
        #critical_mask, trivial_mask = self.cnn(vit_output_unflattened)  # Shape: (B, 1, 16, 16)
        #trivial_mask = 1-critical_mask
        critical_mask, trivial_mask = torch.zeros(B, 1, H, W).cuda(), torch.zeros(B, 1, H, W).cuda()
        vit_l_output_for_critical = (vit_output_unflattened * critical_mask).permute(0, 2, 3, 1).view(B, T, D)  # Shape: (B, 256, 1024)
        vit_l_output_for_trivial = (vit_output_unflattened * trivial_mask).permute(0, 2, 3, 1).view(B, T, D)
        #CAUTION: The above line is for the case where the attention loss is used for the tokens
               
        critical_visual_tokens = self.critical_visual_tokens.repeat(B, 1, 1)
        trivial_visual_tokens = self.trivial_visual_tokens.repeat(B, 1, 1)

        #vit_output_unflattened = vit_l_output.view(B, H, W, D).permute(0, 3, 1, 2)  # Shape: (B, 1024, 16, 16)
        #critical_mask, trivial_mask = self.cnn(vit_output_unflattened)  # Shape: (B, 1, 16, 16)
        #critical_mask, trivial_mask = torch.zeros(B, 1, 16, 16).cuda(), torch.zeros(B, 1, 16, 16).cuda()
        ############################################################################   Option use self-attention for the tokens
        #critical_visual_tokens = self.self_attn_critical(critical_visual_tokens)
        #CAUTION The above line is for the case where the self-attention is used for the tokens
        if self.new_explicd==1:
            agg_critical_visual_tokens, attn_critical_weights = self.cross_attn_critical(critical_visual_tokens, vit_l_output_hsv, vit_l_output)
            #agg_critical_visual_tokens, attn_critical_weights = self.cross_attn(agg_critical_visual_tokens, vit_l_output_hsv, vit_l_output)
            agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial(trivial_visual_tokens, vit_l_output_hsv, vit_l_output)
        elif self.new_explicd==0:
            agg_critical_visual_tokens, attn_critical_weights = self.cross_attn_critical(critical_visual_tokens, vit_l_output, vit_l_output)
            #agg_critical_visual_tokens, attn_critical_weights = self.cross_attn_critical_smaller(agg_critical_visual_tokens, vit_l_output_hsv, vit_l_output)
            agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial(trivial_visual_tokens, vit_l_output, vit_l_output)
            #agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial_smaller(agg_trivial_visual_tokens, vit_l_output, vit_l_output)
        
        d_products = torch.matmul(critical_visual_tokens, vit_l_output_hsv.permute(0, 2, 1))/(D**0.5)
        loss_fn = SpatialAwarePatchLoss_mine_spatial()

          # Compute dot products (before softmax)
        crit_dot = torch.matmul(critical_visual_tokens, vit_l_output_hsv.permute(0, 2, 1))/(D**0.5)  # (7, 256)
        triv_dot = torch.matmul(trivial_visual_tokens, vit_l_output_hsv.permute(0, 2, 1))/(D**0.5)  # (n_t, 256)

        attn_explicd_loss_dict = {}
        if self.config.dataset == 'IDRID' or self.config.dataset == 'IDRID_SOFT' or self.config.dataset == 'IDRID_EDEMA' or self.config.dataset == 'IDRID_EDEMA_SOFT':
            attn_explicd_loss_dict=retinal_loss_function(attn_critical_weights, crit_dot, triv_dot, img_patches.view(B, 3, T))
            attn_explicd_loss_dict["color_loss"]=loss_fn(img_patches, d_products, attn_trivial_weights)["color_loss"]
        elif self.config.dataset == 'ISIC' or self.config.dataset == 'ISIC_SOFT':
            attn_explicd_loss_dict = loss_fn(img_patches, d_products, attn_trivial_weights)
            attn_explicd_loss_dict["var_loss"]=attention_variance_loss(attn_critical_weights)
            attn_explicd_loss_dict["var_loss_other"], attn_explicd_loss_dict["masks"], attn_explicd_loss_dict["crit_dot"]=skin_lesion_loss_function(attn_critical_weights, crit_dot, triv_dot, img_patches.view(B, 3, T))
        
            attn_explicd_loss_dict["smooth_loss"]=smoothness_loss(attn_critical_weights)
            attn_explicd_loss_dict["overlap_loss"]=total_non_overlap_loss(attn_critical_weights, attn_trivial_weights)

        
        attn_explicd_loss_dict["tv_loss"], attn_explicd_loss_dict["laplacian_smoothness_loss"],attn_explicd_loss_dict["entropy"] = attn_chat_gpt_crit_smooth_loss(attn_critical_weights,  merge_patches=False)
        attn_explicd_loss_dict["contrast_loss"], attn_explicd_loss_dict["orthog_loss"],attn_explicd_loss_dict["kl_loss"],attn_explicd_loss_dict["inv_loss"], attn_explicd_loss_dict["dot_loss"],attn_explicd_loss_dict["dist_loss"] = attn_chat_gpt_crit_different_maps(attn_critical_weights, attn_trivial_weights, merge_patches=False)
        #attn_explicd_loss_dict["similarity_loss"] = vectorized_similarity_loss(attn_critical_weights)
        attn_explicd_loss_dict["similarity_loss"] = vectorized_similarity_loss(crit_dot)
        attn_explicd_loss_dict["anchor_alignment_loss"] = anchor_alignment_loss(attn_critical_weights)
        attn_explicd_loss_dict["zero_and_three_loss"] = attention_map_zero_and_three_loss(attn_critical_weights)
        attn_explicd_loss_dict["mutual_exclusivity_loss"] = mutual_exclusivity_loss(attn_critical_weights)
        attn_explicd_loss_dict["high_attention_penalty"] = high_attention_penalty(attn_critical_weights, max_tokens=2)
        attn_explicd_loss_dict["mse_loss"] = mse_loss_between_maps(attn_critical_weights)
        #attn_explicd_loss += 10*gpt4_0_second_attention_loss_mine(attn_critical_weights, attn_trivial_weights)[0]
        #loss_fn = SpatialAwarePatchLoss_mine()
        
        #attn_explicd_loss += 1*loss_fn(patches.permute(0, 2, 1).view(B, D_patch, int(T ** 0.5), int(T ** 0.5)), attn_critical_weights, attn_trivial_weights)
        #attn_explicd_loss += 1*loss_fn(patches.permute(0, 2, 1).view(B, D_patch, int(T ** 0.5), int(T ** 0.5)), d_products, attn_trivial_weights)

        # I've been using this one for skin lesions
        #attn_explicd_loss += 1*loss_fn(hsv_patches, d_products, attn_trivial_weights)

        #attn_explicd_loss += 10000 * torch.mean(attn_critical_weights * attn_trivial_weights)
        #print(f"d_products.shape {d_products.shape}")
        ############################################################################   Option use self-attention for the tokens
        #agg_critical_visual_tokens = self.self_attn_critical(agg_critical_visual_tokens)
        #CAUTION The above line is for the case where the self-attention is used for the tokens
                                                # self.proj(the tokens from X to 512
        agg_critical_visual_tokens_for_explicid = self.proj(self.norm(self.ffn(agg_critical_visual_tokens[:,:self.num_of_criteria,:])))
        agg_critical_visual_tokens_for_explicid = F.normalize(agg_critical_visual_tokens_for_explicid, dim=-1)

        #vit_output_for_Sit = self.norm_vit_output_for_Sit(self.proj_vit_output_for_Sit(vit_l_output))
        #attn_explicd_loss = torch.tensor(0.0, device=self.device)

        overlap_loss = torch.tensor(0.0, device=self.device)
        te_loss = torch.tensor(0.0, device=self.device)
        #overlap_loss = 10000 * torch.mean(attn_critical_weights * attn_trivial_weights)

        #############################################################################
        agg_critical_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn_smaller(agg_critical_visual_tokens)))
        agg_trivial_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn_smaller(agg_trivial_visual_tokens)))

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            ############################### Option dot product
            image_logits_dict[key] = (self.logit_scale *agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            #image_logits_dict[key] = (self.layers[key](image_logits_dict[key])).squeeze(-1)
            ############################### Option cosine similarity
            # norm_agg_visual_tokens = F.normalize(agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :], dim=-1)
            # norm_concept_embedding = F.normalize(self.concept_token_dict[key], dim=-1)
            # cosine_similarity = torch.matmul(norm_agg_visual_tokens, norm_concept_embedding.repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            # image_logits_dict[key] = self.logit_scale * cosine_similarity
            ############################### apply softmax
            #image_logits_dict[key] = F.softmax(image_logits_dict[key], dim=1)
            ############################### DIY entropy
            # temperature=1.0
            # soft_key = F.softmax(image_logits_dict[key]/temperature, dim=-1)
            # te_loss += -0.0001 *(torch.sum(soft_key * torch.log(soft_key)))
            # soft_key_norm = soft_key/soft_key.max(dim=-1)[0].unsqueeze(1)
            # image_logits_dict[key] = soft_key_norm*image_logits_dict[key]

            idx += 1

        image_logits_list = []

        for key in image_logits_dict.keys():
            #image_logits_list.append(F.softmax(image_logits_dict[key], dim=1))
            image_logits_list.append(image_logits_dict[key])

        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits)

        #print(image_logits.shape)  # B, NUM_OF_SIMILARITIES[self.config.dataset]
        cls_with_te_logits = self.cls_wiht_te(image_logits)
        #print(cls_with_te_logits.shape) # B, NUM_OF_CLASSES, 1

        #vit_l_output_for_sit = self.proj_vit_output(self.norm_vit_output(self.ffn_vit_output(vit_l_output)))
        
        to_return_dict = {
            "patches":patches,
            "patches_colored":patches,
            "critical_mask":critical_mask,
            "trivial_mask":trivial_mask,
            "cls_logits":cls_logits,
            "image_logits_dict":image_logits_dict,
            "agg_critical_visual_tokens_for_SiT":F.normalize(agg_critical_visual_tokens_for_SiT, dim=-1),
            "agg_trivial_visual_tokens_for_SiT":F.normalize(agg_trivial_visual_tokens_for_SiT, dim=-1),
            "attn_explicd_loss_dict":attn_explicd_loss_dict,
            "overlap_loss":overlap_loss,
            "attn_critical_weights":attn_critical_weights,
            "attn_trivial_weights":attn_trivial_weights,
            "vit_l_output":vit_l_output,
            "agg_critical_visual_tokens":agg_critical_visual_tokens,
            "agg_trivial_visual_tokens":agg_trivial_visual_tokens,
            "te_loss":te_loss
        }

        doing_cnn_critical = False
        doing_cnn_trivial=False

        to_return_dict["cnn_logits_critical"] = None
        to_return_dict["cnn_logits_trivial"] = None

        if doing_cnn_critical and explicid_imgs_latents is not None:
            to_return_dict["cnn_logits_critical"] = cls_logits
        else:
            to_return_dict["cnn_logits_critical"] = cls_logits

        if doing_cnn_trivial:
            to_return_dict["cnn_logits_trivial"] = self.classifying_trivial_cnn(vit_l_output)

        to_return_dict["cls_with_te_logits"] = cls_with_te_logits.squeeze(-1)
        #to_return_dict["te_loss"] = 0.00001 * te.nn.functional.entropy_logic_loss(self.cls_wiht_te)

        
        # for key in image_logits_dict.keys():
        #     to_return_dict["te_loss"] += 0.00001 * te.nn.functional.entropy_logic_loss(self.layers[key])

        # to_return_dict["te_loss"] += 0.00001 * te.nn.functional.entropy_logic_loss(self.cls_wiht_te)
        return to_return_dict