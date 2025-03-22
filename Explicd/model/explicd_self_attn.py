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
    'ISIC_SOFT': 6,

    'IDRID': 5,
    'IDRID_EDEMA': 6,

    'BUSI': 6,
    'BUSI_SOFT': 6
}

NUM_OF_SIMILARITIES = {
    'ISIC': 34,
    'ISIC_MINE': 24,
    'ISIC_MINIMAL': 49,
    'ISIC_SOFT': 12,

    'IDRID': 18,
    'IDRID_EDEMA': 15,
    
    'BUSI': 17,
    'BUSI_SOFT': 18
}

def get_prefix(task: str, key: str) -> str:
    if task== "ISIC" or task == "ISIC_MINE" or task == "ISIC_MINIMAL" or task == "ISIC_SOFT":
        return f"this is a dermoscopic image, the {key} of the lesion is "
    elif task == "IDRID" or task=='IDRID_EDEMA':
        return f"this is a fundus image, the {key} of the eye is "
    elif task == "BUSI" or task == "BUSI_SOFT":
        return f"this is an ultrasound image of human breasts, the {key} of the tissue is "
    else:
        raise ValueError(f"Unknown task: {task}. Must be either 'derm' or 'us'")
    
def compute_center_of_mass(attn_weights, num_patches=256):
    """
    Calculate the center of mass of the attention map in terms of patch grid.
    Args:
        attn_weights: Attention weights of shape (B, N, 256), where N is number of patches.
        num_patches: Integer representing the size of the patch grid (assumed 16x16).
    Returns:
        center_of_mass: Tensor of shape (B, N, 2), where the last dimension represents (x, y) coordinates.
    """
    B, N, _ = attn_weights.shape  # Attention map shape (Batch, Number of Tokens, Number of Tokens)
    
    patch_grid_size = int(num_patches ** 0.5)  # Since num_patches = 256, patch grid is 16x16
    y_coords, x_coords = torch.meshgrid(torch.arange(patch_grid_size), torch.arange(patch_grid_size))
    coords = torch.stack([x_coords, y_coords], dim=-1).float().to(attn_weights.device)  # Shape: (16, 16, 2)

    coords = coords.view(-1, 2)  # Shape: (256, 2), flattened grid of coordinates

    # Flatten attention weights
    attn_weights_flat = attn_weights.view(B, N, -1)  # Shape: (B, N, 256)

    # Compute the weighted sum of the coordinates to get the center of mass
    weighted_coords = attn_weights_flat @ coords  # Shape: (B, N, 2)
    center_of_mass = weighted_coords / attn_weights_flat.sum(dim=-1, keepdim=True)  # Normalize by sum of weights

    return center_of_mass


def compute_attention_loss(critical_attn_weights, trivial_attn_weights, num_patches, lesion_percentage=0.6):
    """
    Calculate a loss function that:
    - Encourages critical tokens to cluster around the lesion area (top X% attended regions).
    - Ensures trivial tokens focus on non-lesion areas (remaining regions).
    """
    B, N_critical, _ = critical_attn_weights.shape
    _, N_trivial, _ = trivial_attn_weights.shape

    # 1. Compute the overall attention weights across all tokens to get the potential lesion area
    attn_weights_sum = critical_attn_weights.sum(dim=1)  # Sum across critical tokens to get a "global" attention map
    attention_map = attn_weights_sum.view(B, num_patches)  # Shape: (B, num_patches)

    # 2. Create a binary lesion mask based on the top X% of attended patches
    lesion_area = int(lesion_percentage * num_patches)  # Number of patches covering the lesion area
    top_attention_indices = torch.argsort(attention_map, dim=-1, descending=True)[:, :lesion_area]  # Top N patches based on attention

    lesion_mask = torch.zeros((B, num_patches), device=critical_attn_weights.device)
    lesion_mask.scatter_(1, top_attention_indices, 1)  # Set the top X% attended patches to 1 (lesion region)

    # 3. Loss for critical tokens: Penalize attention outside the lesion region (i.e., outside the attended areas)
    critical_masked_attn = critical_attn_weights * lesion_mask.unsqueeze(1)
    critical_mask_loss = critical_masked_attn.sum(dim=-1) / critical_attn_weights.sum(dim=-1)

    # 4. Loss for trivial tokens: Penalize attention inside the lesion region (i.e., overlap with lesion area)
    trivial_masked_attn = trivial_attn_weights * (1 - lesion_mask.unsqueeze(1))
    trivial_mask_loss = trivial_masked_attn.sum(dim=-1) / trivial_attn_weights.sum(dim=-1)

    # 5. Center of Mass Loss: Encourage critical tokens to focus on similar regions (cluster)
    critical_centers_of_mass = compute_center_of_mass(critical_attn_weights, num_patches)
    
    # Calculate pairwise distances between centers of mass of critical tokens
    critical_com_distance = torch.norm(critical_centers_of_mass[:, :, None, :] - critical_centers_of_mass[:, :, :, None], dim=-1)  # Pairwise distances between centers
    clustering_loss = critical_com_distance.mean()  # Encourage smaller distances (i.e., clustering)

    # Combine all loss terms
    total_loss = critical_mask_loss.mean() + trivial_mask_loss.mean() + clustering_loss

    return total_loss

def gpt4_0_attention_loss(attn_critical_weights, attn_trivial_weights):
    # Clustering loss for critical tokens
    B, N_crit, T = attn_critical_weights.shape
    attn_critical_weights = attn_critical_weights.view(B, N_crit, -1)
    cluster_loss = 0
    for i in range(N_crit):
        for j in range(i + 1, N_crit):
            cluster_loss += F.mse_loss(attn_critical_weights[:, i, :], attn_critical_weights[:, j, :])
    cluster_loss /= (N_crit * (N_crit - 1) / 2)

    # Dispersion loss for trivial tokens
    B, N_triv, T = attn_trivial_weights.shape
    attn_trivial_weights = attn_trivial_weights.view(B, N_triv, -1)
    dispersion_loss = 0
    for i in range(N_triv):
        for j in range(i + 1, N_triv):
            dispersion_loss += -F.mse_loss(attn_trivial_weights[:, i, :], attn_trivial_weights[:, j, :])
    dispersion_loss /= (N_triv * (N_triv - 1) / 2)

    # Combine the losses
    total_loss = cluster_loss + dispersion_loss
    return total_loss

def sonnet_compute_attention_loss(attn_critical, attn_trivial):
    """
    Args:
        attn_critical: (B,7,256) attention weights
        attn_trivial: (B,14,256) attention weights
    """
    B = attn_critical.shape[0]
    
    # Aggregate attention maps
    critical_map = attn_critical.sum(dim=1)  # (B,256)
    trivial_map = attn_trivial.sum(dim=1)   # (B,256)
    
    # Reshape to spatial dimensions
    critical_spatial = critical_map.view(B, 16, 16)
    
    # 1. Spatial Continuity Loss for critical attention
    dx = torch.abs(critical_spatial[:, :, 1:] - critical_spatial[:, :, :-1])
    dy = torch.abs(critical_spatial[:, 1:, :] - critical_spatial[:, :-1, :])
    continuity_loss = (dx.mean() + dy.mean()) / 2.0
    
    # 2. Coverage Loss
    coverage = critical_map.mean(dim=1)  # (B,)
    coverage_loss = F.relu(coverage - 0.6).mean()
    
    # 3. Separation Loss
    overlap = (critical_map * trivial_map).sum(dim=1).mean()
    separation_loss = overlap
    
    # 4. Uniformity Loss
    critical_std = torch.std(attn_critical, dim=1).mean()
    trivial_std = torch.std(attn_trivial, dim=1).mean()
    uniformity_loss = critical_std + F.relu(0.1 - trivial_std)
    
    # Combine losses with weights
    total_loss = (
        2.0 * continuity_loss +
        1.0 * coverage_loss + 
        1.0 * separation_loss +
        0.5 * uniformity_loss
    )
    
    return total_loss


def gpt4_0_second_attention_loss(attn_critical, attn_non_critical,  cluster_sigma, orthogonal_sigma, coverage_sigma):
    """
    Computes attention loss to enforce structured attention behavior.
    
    Parameters:
    - attn_critical: (B, 7, 256) Attention maps of critical tokens.
    - attn_non_critical: (B, 14, 256) Attention maps of non-critical tokens.
    - lambda1, lambda2, lambda3: Hyperparameters for loss terms.
    """

    cluster_sigma = F.softplus(cluster_sigma) + 1e-6
    orthogonal_sigma = F.softplus(orthogonal_sigma) + 1e-6
    coverage_sigma = F.softplus(coverage_sigma) + 1e-6
    
    # --- Clustering Loss (Encouraging smooth & compact attention for critical tokens) ---
    variance = torch.var(attn_critical, dim=-1).mean()  # Variance across spatial dimension
    cluster_loss = variance  # Lower variance = more compact focus
    
    # --- Orthogonality Loss (Ensuring non-critical tokens avoid critical regions) ---
    attn_critical_norm = F.normalize(attn_critical, p=2, dim=-1)
    attn_non_critical_norm = F.normalize(attn_non_critical, p=2, dim=-1)
    orthog_loss = (attn_critical_norm @ attn_non_critical_norm.transpose(1, 2)).pow(2).mean()

    #orthog_loss = (attn_critical @ attn_non_critical.transpose(1, 2)).pow(2).mean()
    
    # --- Coverage Loss (Ensuring all areas are attended to) ---
    total_attention = attn_critical.sum(dim=1)/2 + attn_non_critical.sum(dim=1)/2  # (B, 256)
    #total_attention = total_attention / total_attention.max(dim=-1, keepdim=True)[0]  # Normalize

    coverage_loss = F.mse_loss(total_attention, torch.ones_like(total_attention))

    # total_attention = attn_critical.sum(dim=1) + attn_non_critical.sum(dim=1)  # (B, 256)
    # coverage_loss = F.mse_loss(total_attention, torch.ones_like(total_attention))
    
    # --- Final Loss Computation ---
    # total_loss = (
    #     lambda1 * cluster_loss + 
    #     lambda2 * orthog_loss + 
    #     lambda3 * coverage_loss
    # )

    # print(f"cluster loss {cluster_loss}")
    # print(f"orthog_loss loss {orthog_loss}")
    # print(f"coverage_loss loss {coverage_loss}")
    cluster_scale = 0
    orthogonal_scale = 1.5
    coverage_scale = 1.5


    # total_loss = cluster_scale*(cluster_loss / (2 * cluster_sigma**2)) + orthogonal_scale*(orthog_loss / (2 * orthogonal_sigma**2)) + coverage_scale*(coverage_loss / (2 * coverage_sigma**2)) \
    #             + torch.log(cluster_sigma) + torch.log(orthogonal_sigma) + torch.log(coverage_sigma)

    log_weight = 0.1  # Reduce impact of log terms
    total_loss = (
        cluster_scale * (cluster_loss / (2 * cluster_sigma**2)) +
        orthogonal_scale * (orthog_loss / (2 * orthogonal_sigma**2)) +
        coverage_scale * (coverage_loss / (2 * coverage_sigma**2)) +
        log_weight * (torch.log(cluster_sigma) + torch.log(orthogonal_sigma) + torch.log(coverage_sigma))
    )
    # You can assign weights to prioritize certain losses (optional)
    # w1, w2, w3 = 0.5, 1.0, 1.0  # Example weights for each loss term

    # # Combine the losses with the weights and scaled values
    # total_loss = (
    #     w1 * cluster_loss + 
    #     w2 * orthog_loss + 
    #     w3 * coverage_loss
    # )

    # print(f"cluster loss {inverse_scaled_loss1}")
    # print(f"orthog_loss loss {inverse_scaled_loss2}")
    # print(f"coverage_loss loss {inverse_scaled_loss3}")

    return total_loss, {
        "cluster_loss": cluster_loss.item(),
        "orthog_loss": orthog_loss.item(),
        "coverage_loss": coverage_loss.item()
    }

def gpt4_0_second_attention_loss_mine(attn_critical, attn_non_critical):
    """
    Computes attention loss to enforce structured attention behavior.
    
    Parameters:
    - attn_critical: (B, 7, 256) Attention maps of critical tokens.
    - attn_non_critical: (B, 14, 256) Attention maps of non-critical tokens.
    - lambda1, lambda2, lambda3: Hyperparameters for loss terms.
    """
    
    _, _, num_patches = attn_critical.shape

    # --- Clustering Loss (Encouraging smooth & compact attention for critical tokens) ---
    variance = torch.var(attn_critical, dim=-1).mean()  # Variance across spatial dimension
    cluster_loss = variance  # Lower variance = more compact focus
    
    # --- Orthogonality Loss (Ensuring non-critical tokens avoid critical regions) ---
    attn_critical_norm = F.normalize(attn_critical, p=2, dim=-1)
    attn_non_critical_norm = F.normalize(attn_non_critical, p=2, dim=-1)
    orthog_loss = (attn_critical_norm @ attn_non_critical_norm.transpose(1, 2)).pow(2).mean()

    #orthog_loss = (attn_critical @ attn_non_critical.transpose(1, 2)).pow(2).mean()
    ################################### Option normalize by diving by max value
    attn_critical = attn_critical / (attn_critical.max(dim=-1, keepdim=True).values)
    attn_non_critical = attn_non_critical / (attn_non_critical.max(dim=-1, keepdim=True).values)
    total_attention = attn_critical + attn_non_critical  # (B, 7, 256)
    coverage_loss = F.mse_loss(total_attention, torch.ones_like(total_attention))
    ################################### Option sum um halfs
    # --- Coverage Loss (Ensuring all areas are attended to) ---
    # total_attention = attn_critical/2 + attn_non_critical/2  # (B, 7, 256)
    # #total_attention = total_attention / total_attention.max(dim=-1, keepdim=True)[0]  # Normalize
    # coverage_loss = F.mse_loss(total_attention, torch.ones_like(total_attention)/num_patches)
    ###################################

    cluster_scale = 0
    orthogonal_scale = 1.5
    coverage_scale = 1.5


    total_loss = (
        cluster_scale * cluster_loss  +
        orthogonal_scale * orthog_loss  +
        coverage_scale * coverage_loss
    )
    # You can assign weights to prioritize certain losses (optional)
    # w1, w2, w3 = 0.5, 1.0, 1.0  # Example weights for each loss term

    # # Combine the losses with the weights and scaled values
    # total_loss = (
    #     w1 * cluster_loss + 
    #     w2 * orthog_loss + 
    #     w3 * coverage_loss
    # )

    # print(f"cluster loss {inverse_scaled_loss1}")
    # print(f"orthog_loss loss {inverse_scaled_loss2}")
    # print(f"coverage_loss loss {inverse_scaled_loss3}")

    return total_loss, {
        "cluster_loss": cluster_loss.item(),
        "orthog_loss": orthog_loss.item(),
        "coverage_loss": coverage_loss.item()
    }

class AdaptiveCriticalPatchLoss(nn.Module):
    def __init__(self, alpha=1.0):
        """
        Dynamically adjusts loss weights based on mask coverage.

        Args:
            alpha (float): Controls how aggressively λ₁ and λ₂ adjust.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, image_feats, mask):
        """
        Args:
            image_feats (torch.Tensor): Image feature map (B, 1024, 16, 16).
            mask (torch.Tensor): Binary segmentation mask (B, 1, 16, 16), values ~ 0 or 1.

        Returns:
            torch.Tensor: Combined loss.
        """
        B, C, H, W = image_feats.shape  # C = 1024, H = 16, W = 16
        num_patches = H * W

        # Flatten patches (B, C, 256) and mask (B, 1, 256)
        image_feats = image_feats.view(B, C, num_patches)
        mask = mask.view(B, 1, num_patches)

        # Compute mask coverage (fraction of critical patches)
        mask_coverage = mask.mean(dim=2, keepdim=True)  # (B, 1, 1)

        # Adaptive scaling of λ₁ and λ₂
        lambda_within = torch.exp(-self.alpha * mask_coverage)  # If mask too small, increase λ₁
        lambda_between = 4*torch.exp(self.alpha * mask_coverage)  # If mask too large, increase λ₂

        # Extract critical and trivial patches
        critical_patches = image_feats * mask
        trivial_patches = image_feats * (1 - mask)

        # Normalize for cosine similarity
        critical_patches = F.normalize(critical_patches, p=2, dim=1)
        trivial_patches = F.normalize(trivial_patches, p=2, dim=1)

        # Loss 1: Encourage similarity among critical patches
        similarity_critical = torch.bmm(critical_patches.transpose(1, 2), critical_patches)  # (B, 256, 256)
        loss_within = (1 - similarity_critical.mean()) * lambda_within.mean()

        # Loss 2: Discourage similarity between critical & trivial patches
        similarity_between = torch.bmm(critical_patches.transpose(1, 2), trivial_patches)  # (B, 256, 256)
        loss_between = similarity_between.mean() * lambda_between.mean()

        # Total loss
        total_loss = loss_within + loss_between

        return total_loss


class SpatialAwarePatchLoss(nn.Module):
    def __init__(self, gamma=5.0, sigma=0.2):
        """
        Args:
            gamma (float): Controls how strongly distance affects penalties.
            sigma (float): Controls the sharpness of spatial weighting.
        """
        super().__init__()
        self.gamma = gamma
        self.sigma = sigma

    def compute_distance_matrix(self, H, W, device):
        """Compute normalized spatial distance matrix for a grid (H, W)."""
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1)  # (H, W, 2)
        coords = coords.reshape(-1, 2).to(device)  # (256, 2) for a 16x16 grid

        # Compute pairwise Euclidean distance
        dist_matrix = torch.cdist(coords.float(), coords.float(), p=2)  # (256, 256)

        # Normalize to [0,1] using the largest distance in the grid
        dist_matrix /= dist_matrix.max()
        return dist_matrix  # (256, 256)

    def forward(self, image_feats, mask):
        """
        Args:
            image_feats (torch.Tensor): Image features (B, 1024, 16, 16).
            mask (torch.Tensor): Binary segmentation mask (B, 1, 16, 16).

        Returns:
            torch.Tensor: Spatially aware loss.
        """
        B, C, H, W = image_feats.shape
        num_patches = H * W
        device = image_feats.device

        # Flatten patches (B, 1024, 256) and mask (B, 1, 256)
        image_feats = image_feats.view(B, C, num_patches)
        mask = mask.view(B, 1, num_patches)
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


        # Extract critical and trivial patches
        critical_patches = image_feats * mask  # (B, 1024, 256) * (B, 1, 256)
        trivial_patches = image_feats * (1 - mask)  # (B, 1024, 256) * (B, 1, 256)

        # Normalize features for cosine similarity
        critical_patches = F.normalize(critical_patches, p=2, dim=1)
        trivial_patches = F.normalize(trivial_patches, p=2, dim=1)

        # Compute similarity matrices
        similarity_critical = torch.bmm(critical_patches.transpose(1, 2), critical_patches)  # (B, 256, 256)
        similarity_between = torch.bmm(critical_patches.transpose(1, 2), trivial_patches)  # (B, 256, 256)

        # Loss 1: Critical patches should be similar if spatially close
        loss_within = (((similarity_critical - 1) ** 2 * spatial_weight)*lambda_within).mean()

        # Loss 2: Trivial patches should be dissimilar to nearby critical patches
        loss_between = ((similarity_between ** 2 * spatial_weight)*lambda_between).mean()

        # Total loss
        total_loss = loss_within + loss_between
        return total_loss

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

        weighted_feat_sim = feat_sim*normalized_D[None, :, :].to(device)
        
        #weighted_feat_sim = feat_sim*D[None, :, :].to(device)
        #weighted_feat_sim = feat_sim
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
                loss_sim = F.mse_loss(attn_sim_crit, weighted_feat_sim)
                #loss_sim = alpha*F.mse_loss(attn_sim_crit, weighted_feat_sim) + (1-alpha)*F.mse_loss(attn_sim_crit, feat_sim)
                total_loss += 10 * loss_sim
            else:
                total_loss += 10* F.mse_loss(attn_critical[:, 0, :], attn_critical[:, i, :])

        # Encourage smoothness in attention maps (low variance)
        # mean_critical_attention = attn_critical.mean(dim=1, keepdim=True)  # (B, 1, 256)
        # variance = ((attn_critical - mean_critical_attention) ** 2).mean(dim=1)  # (B, 256)
        # total_loss += variance.mean()

        return total_loss

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
        black_patch_mask = (saturation < threshold) & (value < threshold)  # Shape (B, 256)
        
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
    
    # Enforce that critical tokens match expected color patches
    L_color = F.mse_loss(crit_dot[:,:crit_dot.shape[1],:], M, reduction="none")  # Compare first 5 critical tokens to their expected maps
    #print(f"L color shape {L_color.shape}")
    attn_loss_dict["color_loss"] = L_color
    #L_color = 1 - F.cosine_similarity(crit_dot[:,:crit_dot.shape[1],:], M, dim=-1).mean()
    
    # 3) Automatic Trivial Token Reduction (Minimize their dot products)
    L_sparsity = torch.mean(torch.abs(triv_dot))
    attn_loss_dict["sparsity_loss"] = L_sparsity
    
    # Final loss
   # loss = lambda1 * L_overlap + lambda2 * L_color + lambda3 * L_sparsity
    
    return attn_loss_dict

# class SpatialAwarePatchAttentionLoss(nn.Module):
#     def __init__(self, H=16, W=16, sigma=2.0):
#         super().__init__()
#         self.H, self.W = H, W
#         self.sigma = sigma  # Controls locality strength
        
#         # Compute spatial distances once and store
#         self.register_buffer("spatial_weight", self.compute_distance_matrix(H, W))

#     def compute_distance_matrix(self, H, W):
#         """Compute spatial distance matrix between patches in a 16x16 grid."""
#         grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
#         coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).float()  # (256, 2)

#         # Compute pairwise Euclidean distance
#         dist_matrix = torch.cdist(coords, coords, p=2)  # (256, 256)
        
#         # Convert distance to weight (closer = higher weight)
#         spatial_weight = torch.exp(-dist_matrix / self.sigma).to(device="cuda")
#         return spatial_weight  # (256, 256)

#     def forward(self, image_feats, critical_weights, trivial_weights):
#         """
#         image_feats: (B, 1024, 16, 16)   -> Image features (input to ViT)
#         critical_tokens: (B, 7, 1024)    -> Critical region token embeddings
#         trivial_tokens: (B, 14, 1024)    -> Trivial region token embeddings
#         critical_weights: (B, 7, 256)    -> Attention maps for critical tokens (soft mask)
#         trivial_weights: (B, 14, 256)    -> Attention maps for trivial tokens (soft mask)
#         """

#         B, C, H, W = image_feats.shape
#         _, num_critical, _, _ = critical_weights.shape  # (B, 7, 16, 16)
#         num_trivial = trivial_weights.shape[1]
#         image_feats = image_feats.flatten(2).transpose(1, 2)  # (B, 256, 1024)

#         # Compute feature similarity matrix using cosine similarity
#         feat_sim = F.cosine_similarity(image_feats.unsqueeze(2), image_feats.unsqueeze(1), dim=-1)  # (B, 256, 256)

#         # Combine feature similarity with spatial weight
#         combined_weight = torch.exp(-self.spatial_weight) * feat_sim  # (B, 256, 256)

#         # Compute intra-critical similarity (should be high)
#         #print(f"critical_weights shape {critical_weights.shape}")
#         #print(f"combined_weight shape {combined_weight.shape}")
#         critical_spatial_weights = torch.bmm(critical_weights.view(B, num_critical, H * W), combined_weight)  # (B, 7, 256)
#         critical_spatial_weights = torch.bmm(critical_spatial_weights, critical_weights.view(B, num_critical, H * W).transpose(1, 2))  # (B, 7, 7)

#         loss_within = torch.mean(1 - critical_spatial_weights)  # Minimize within-mask dissimilarity

#         # Compute cross-mask similarity (should be low)
#         cross_spatial_weights = torch.bmm(critical_weights.view(B, num_critical, H * W), combined_weight)  # (B, 7, 256)
#         cross_spatial_weights = torch.bmm(cross_spatial_weights, trivial_weights.view(B, num_trivial, H * W).transpose(1, 2))  # (B, 7, 14)

#         loss_between = torch.mean(cross_spatial_weights)  # Maximize critical/trivial dissimilarity

#         # Balance losses (avoid collapse to a single mask)
#         lambda_within = 1.0  # Weight for making critical patches more similar
#         lambda_between = 0.5  # Weight for separating critical/trivial
        
#         loss = lambda_within * loss_within + lambda_between * loss_between
#         return loss

class SpatialAwarePatchAttentionLoss(nn.Module):
    def __init__(self, H=16, W=16, sigma=2.0):
        super().__init__()
        self.H, self.W = H, W
        self.sigma = sigma  # Controls locality strength
        self.register_buffer("spatial_weight", self.compute_distance_matrix(H, W))

    def compute_distance_matrix(self, H, W):
        grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).float()
        dist_matrix = torch.cdist(coords, coords, p=2)
        return torch.exp(-dist_matrix / self.sigma).to(device="cuda")

    def compute_mean_color_similarity(self, image_feats):
        """Compute mean color similarity for patches."""
        mean_color = image_feats.mean(dim=-1, keepdim=True)  # (B, 1, 256)
        return 1 - torch.cdist(mean_color, mean_color, p=2)  # (B, 256, 256)

    def compute_texture_similarity(self, image_feats):
        """Compute texture similarity using Laplacian variance."""
        laplacian = T.GaussianBlur(3, sigma=(1.5, 1.5))(image_feats) - image_feats
        texture_variance = laplacian.var(dim=-1, keepdim=True)  # (B, 1, 256)
        return 1 - torch.cdist(texture_variance, texture_variance, p=2)  # (B, 256, 256)

    def forward(self, image_feats, critical_weights, trivial_weights):
        B, C, H, W = image_feats.shape
        #critical_weights = torch.sigmoid(critical_weights / 0.005)
        #trivial_weights = torch.sigmoid(trivial_weights / 0.005)
        num_critical = critical_weights.shape[1]
        num_trivial = trivial_weights.shape[1]
        image_feats = image_feats.flatten(2).transpose(1, 2)  # (B, 256, 1024)
        
        # Compute different similarity matrices
        feat_sim = F.cosine_similarity(image_feats.unsqueeze(2), image_feats.unsqueeze(1), dim=-1)  # (B, 256, 256)
        mean_color_sim = self.compute_mean_color_similarity(image_feats)
        texture_sim = self.compute_texture_similarity(image_feats)
        
        # Combine feature similarity with spatial weight
        combined_weight = torch.exp(-self.spatial_weight) * feat_sim  # (B, 256, 256)
        #print(f"combined_weight shape {combined_weight.shape}")
        combined_weight = torch.zeros_like(combined_weight)
        
        # Apply custom similarity measures to specific tokens
        combined_weight[:, 0, :] = mean_color_sim[:, 0, :]
        combined_weight[:, :, 0] = mean_color_sim[:, :, 0]
        #print(f"mean_color_sim shape {mean_color_sim.shape}")
        #print(f"texture_sim shape {texture_sim.shape}")
        combined_weight[:, 4, :] = texture_sim[:, 4, :]
        combined_weight[:, :, 4] = texture_sim[:, :, 4]
        
        # Compute intra-critical similarity (should be high)
        critical_spatial_weights = torch.bmm(critical_weights.view(B, num_critical, H * W), combined_weight)
        critical_spatial_weights = torch.bmm(critical_spatial_weights, critical_weights.view(B, num_critical, H * W).transpose(1, 2))
        loss_within = torch.mean(1 - critical_spatial_weights)
        
        # Compute cross-mask similarity (should be low)
        cross_spatial_weights = torch.bmm(critical_weights.view(B, num_critical, H * W), combined_weight)
        cross_spatial_weights = torch.bmm(cross_spatial_weights, trivial_weights.view(B, num_trivial, H * W).transpose(1, 2))
        loss_between = torch.mean(cross_spatial_weights)
        
        overlap = (critical_weights[:,0,:] * trivial_weights[:,0,:]).sum(dim=1).mean() +(critical_weights[:,4,:] * trivial_weights[:,4,:]).sum(dim=1).mean()
        
        # Balance losses
        loss = loss_within + 0.5 * loss_between + overlap
        return loss


class PatchSelectorCNN(nn.Module):
    def __init__(self):
        super(PatchSelectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, padding=1)
        self.conv_conscise = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.split_1 = torch.nn.Parameter(torch.ones(512,2))
        self.split_2 = torch.nn.Parameter(torch.ones(128,2))

    def forward(self, x):
        #################################### Option cascade of cnn
        x = F.relu(self.conv1(x))
        #softmax_output = F.softmax(self.split_1 / temperature, dim=-1)
        # print(f" softmax_output[:,0] shape { softmax_output[:,0].shape}")
        # x_critical = x * softmax_output[:,0].view(1, 512, 1, 1)
        # x_trivial = x * softmax_output[:,1].view(1, 512, 1, 1)
        x = F.relu(self.conv2(x))
        # softmax_output = F.softmax(self.split_2 / temperature, dim=-1)
        # print(f" softmax_output[:,0] shape { softmax_output[:,0].shape}")
        # x_critical = x_critical * softmax_output[:,0].view(1, 512, 1, 1)
        # x_trivial = x_trivial * softmax_output[:,1].view(1, 512, 1, 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        # x = self.sigmoid(self.conv4(x)
        #################################### Option conscise
        #x = self.conv_conscise(x)
        ####################################
        temperature = 0.0005  # Low temperature (small value makes softmax more peaked)
        scaled_logits = x / temperature
        #print(f"x shape {x.shape}")
        softmax_output = F.softmax(scaled_logits, dim=1)
        x_critical_mask =softmax_output[:,0,:, :].unsqueeze(1)
        x_trivial_mask = softmax_output[:,1,:, :].unsqueeze(1)
        #x_trivial = 1-x_critical
        # return x_critical, x_trivial, softmax_output
        #x = self.sigmoid(x)
        #print(f"x_critical shape {x_critical.shape}")
        #print(f"x_trivial shape {x_trivial.shape}")
        return x_critical_mask, x_trivial_mask


class PatchSelectorCNN_kernel_14(nn.Module):
    def __init__(self, temperature=0.1):
        super(PatchSelectorCNN_kernel_14, self).__init__()
        self.temperature = temperature
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=14, stride=14, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Conv2d(512, 2, kernel_size=1)  # 2 channels: critical/trivial
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)  # Output shape: (batch, 2, H, W)
        
        # Apply softmax with low temperature
        x = F.softmax(x / self.temperature, dim=1)

        x_critical_mask =x[:,0,:, :].unsqueeze(1)
        x_trivial_mask = x[:,1,:, :].unsqueeze(1)

        return x_critical_mask, x_trivial_mask

class PatchSelectorCNNConscise(nn.Module):
    def __init__(self):
        super(PatchSelectorCNNConscise, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=3, padding=1)

        self.conscise = nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x):
        #################################### Option 2 layers
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        #################################### Option 1 layer
        #x = self.conscise(x)
        ####################################
        temperature = 0.05  # Low temperature (small value makes softmax more peaked)
        scaled_logits = x / temperature
        #print(f"x shape {x.shape}")
        softmax_output = F.softmax(scaled_logits, dim=1)
        x_critical_mask =softmax_output[:,0,:, :].unsqueeze(1)
        x_trivial_mask = softmax_output[:,1,:, :].unsqueeze(1)
        return x_critical_mask, x_trivial_mask

class ClassifyingCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(ClassifyingCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Assuming input size is (16, 16) and kernel size is 3
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        self.feature_maps = F.relu(self.conv4(x))  # Save feature maps for Grad-CAM
        x = self.feature_maps.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def guided_backpropagation(model, input_tensor, target_class):
    model.eval()
    input_tensor.requires_grad = True

    output = model(input_tensor)
    model.zero_grad()
    target = output[0][target_class]
    target.backward()

    guided_gradients = input_tensor.grad.data
    return guided_gradients

# def grad_cam(model, input_tensor, target_class):
#     model.eval()
#     output = model(input_tensor)
#     model.zero_grad()
#     target = output[0][target_class]
#     target.backward()

#     gradients = model.feature_maps.grad.data
#     pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
#     feature_maps = model.feature_maps.data

#     for i in range(feature_maps.shape[1]):
#         feature_maps[:, i, :, :] *= pooled_gradients[i]

#     heatmap = torch.mean(feature_maps, dim=1).squeeze()
#     heatmap = np.maximum(heatmap.cpu().numpy(), 0)
#     heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[3]))
#     heatmap = heatmap / np.max(heatmap)
#     return heatmap

# def guided_grad_cam(model, input_tensor, target_class):
#     guided_gradients = guided_backpropagation(model, input_tensor, target_class)
#     cam = grad_cam(model, input_tensor, target_class)

#     guided_grad_cam = guided_gradients[0].cpu().numpy() * cam[..., np.newaxis]
#     return guided_grad_cam

def gaussian_kernel(x, y, sigma=1.0):
    """
    Computes the Gaussian (RBF) kernel between two sets of points.
    x, y: tensors of shape (B, 1, 256)
    """
    diff = x - y
    dist_sq = torch.sum(diff**2, dim=-1)
    return torch.exp(-dist_sq / (2 * sigma**2))

def mmd_loss(tensor1, tensor2, sigma=1.0):
    """
    Computes Maximum Mean Discrepancy (MMD) loss between two tensors.
    tensor1, tensor2: Shape (B, 1, 256)
    """
    # Flatten the tensors for pairwise kernel computation
    tensor1 = tensor1.view(tensor1.size(0), -1)  # Shape: (B, 256)
    tensor2 = tensor2.view(tensor2.size(0), -1)  # Shape: (B, 256)

    # Compute pairwise MMD loss using a Gaussian kernel
    xx = gaussian_kernel(tensor1, tensor1, sigma)
    yy = gaussian_kernel(tensor2, tensor2, sigma)
    xy = gaussian_kernel(tensor1, tensor2, sigma)

    # MMD loss
    return torch.mean(xx + yy - 2 * xy)

def center_of_mass_distance_loss(critical_centers_of_mass, trivial_centers_of_mass):
    """
    Calculates the loss that incentivizes large distances between the critical and trivial token centers of mass.
    
    Args:
        critical_centers_of_mass: Tensor of shape (B, num_critical_tokens, 2) - Centers of mass for critical tokens
        trivial_centers_of_mass: Tensor of shape (B, num_trivial_tokens, 2) - Centers of mass for trivial tokens
        
    Returns:
        loss: A scalar tensor representing the loss
    """
    ##############################################################################################################
    # Compute pairwise distances between critical and trivial centers of mass
    # The shape of the pairwise distance matrix will be (B, num_critical_tokens, num_trivial_tokens)
    # critical_centers_of_mass_expanded = critical_centers_of_mass.unsqueeze(2)  # Shape: (B, num_critical_tokens, 1, 2)
    # trivial_centers_of_mass_expanded = trivial_centers_of_mass.unsqueeze(1)  # Shape: (B, 1, num_trivial_tokens, 2)

    # # Calculate the pairwise Euclidean distance between critical and trivial centers of mass
    # pairwise_distances = torch.norm(critical_centers_of_mass_expanded - trivial_centers_of_mass_expanded, dim=-1)

    # # We want to incentivize large distances, so we take the minimum distance
    # min_distance = pairwise_distances.min(dim=-1).values  # Min distance for each critical token
    
    # # Normalize the loss to be in a manageable range (divide by max possible distance)
    # max_distance = torch.max(pairwise_distances)  # Maximum distance in the batch
    # loss = -min_distance.mean() / max_distance  # Normalize the loss
    ##############################################################################################################
    margin=0.5
    pairwise_distances = torch.norm(critical_centers_of_mass.unsqueeze(2) - trivial_centers_of_mass.unsqueeze(1), dim=-1)
    loss = torch.mean(torch.clamp(margin - pairwise_distances, min=0))  # Margin for separation

    return loss


def gaussian_distribution(size, mean=128, std=40):
    """
    Generate a Gaussian distribution centered at `mean` with standard deviation `std`.
    Size is the length of the distribution (256 in this case).
    """
    x = torch.arange(size).float()
    gauss = torch.exp(-((x - mean)**2) / (2 * std**2))
    return gauss / gauss.sum()  # Normalize to make it a probability distribution

def gaussian_attention_loss(critical_attn_map, trivial_attention_map, size=256, mean=128, std=40):
    """
    Compute the loss that encourages the attention maps to resemble a flattened Gaussian distribution.
    """
    # Generate the target flattened Gaussian distribution
    target_gaussian = gaussian_distribution(size, mean, std).unsqueeze(0).unsqueeze(0).to("cuda") # Shape (1, 1, 256)

    # The attention map has shape (B, 7, 256), we want each attention map to match the Gaussian distribution
    # We compute the MSE between each token's attention map and the target Gaussian distribution.
    critical_loss = 0
    trivial_loss = 0
    for b in range(critical_attn_map.size(0)):  # Iterate over batch
        for t in range(critical_attn_map.size(1)):  # Iterate over each token
            token_attention = critical_attn_map[b, t]  # Shape (256)
            
            # Compute the Mean Squared Error (MSE) between the attention and the Gaussian distribution
            mse_loss = F.mse_loss(token_attention, target_gaussian.squeeze(0).squeeze(0))
            critical_loss += mse_loss

    for b in range(trivial_attention_map.size(0)):  # Iterate over batch
        for t in range(trivial_attention_map.size(1)):  # Iterate over each token
            token_attention = trivial_attention_map[b, t]  # Shape (256)
    
            mse_loss = F.mse_loss(token_attention, target_gaussian.squeeze(0).squeeze(0))
            # To make the loss the opposite, we subtract the MSE from 1
            trivial_loss += (1 - mse_loss)
            
    return critical_loss / (critical_attn_map.size(0) * critical_attn_map.size(1)), trivial_loss / (trivial_attention_map.size(0) * trivial_attention_map.size(1))  # Normalize by batch size and number of tokens


def compute_16x16_loss(X, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0):
    """
    X: Tensor of shape (B, 1, 16, 16)
    lambda1: Weight for smoothness loss
    lambda2: Weight for compactness loss
    lambda3: Weight for background suppression loss
    lambda4: Weight for non-uniformity loss
    """
    B, C, H, W = X.shape
    assert C == 1, "Input tensor should have a single channel"
    X = X.squeeze(1)  # Shape (B, 16, 16)
    
    # Smoothness loss (Encourages clustering by minimizing differences between neighbors)
    dx = torch.abs(X[:, :-1, :] - X[:, 1:, :])
    dy = torch.abs(X[:, :, :-1] - X[:, :, 1:])
    loss_smooth = (dx.mean() + dy.mean())
    
    # Compute the weighted center of mass (μ_x, μ_y)
    i_coords = torch.arange(H, device=X.device).view(1, H, 1).expand(B, H, W)
    j_coords = torch.arange(W, device=X.device).view(1, 1, W).expand(B, H, W)
    
    total_mass = X.sum(dim=[1,2], keepdim=True) + 1e-6  # Avoid division by zero
    mu_x = (i_coords * X).sum(dim=[1,2], keepdim=True) / total_mass
    mu_y = (j_coords * X).sum(dim=[1,2], keepdim=True) / total_mass
    
    # Compactness loss (Encourages a single cluster around the center of mass)
    distances = (i_coords - mu_x).pow(2) + (j_coords - mu_y).pow(2)
    loss_compact = (X * distances).mean()
    
    # Background suppression loss (Encourages values outside the cluster to be small)
    loss_background = (X * (distances > (H//4)**2)).pow(2).mean()
    
    # Non-uniformity loss (Encourages variation in pixel values)
    #loss_uniform = -torch.var(X, dim=[1,2]).mean()
    loss_uniform = torch.var(X, dim=[1,2]).mean()

    smooth_scale=10e2
    compact_scale=10e1
    background_scale=10e2
    uniform_scale=0

    loss_smooth = smooth_scale * loss_smooth
    loss_compact = compact_scale * loss_compact
    loss_background = background_scale * loss_background
    loss_uniform = uniform_scale * loss_uniform
    
    # print(f"loss_smooth {loss_smooth}")
    # print(f"loss_compact {loss_compact}")
    # print(f"loss_background {loss_background}")
    # print(f"loss_uniform {loss_uniform}")

    # Total loss
    loss = (lambda1 * loss_smooth +
            lambda2 * loss_compact +
            lambda3 * loss_background +
            lambda4 * loss_uniform)
    
    return loss

def smarter_attention_loss(attn_critical, attn_trivial, critical_mask=None, trivial_mask=None):
    """
    Computes the attention loss to enforce structured attention behavior.
    
    Parameters:
    - attn_critical: (B, 7, 256) Attention maps of critical tokens.
    - attn_trivial: (B, 14, 256) Attention maps of trivial tokens.
    """
    B, N_crit, T = attn_critical.shape
    _, _, _ = attn_trivial.shape

    ######################################################################### Option with distance for clustering
       # --- Clustering Loss (Encouraging critical tokens to focus on similar regions) ---
    attn_critical_flat = attn_critical.view(B, N_crit, -1)
    cluster_loss = 0
    for i in range(N_crit):
        for j in range(i + 1, N_crit):
            cluster_loss += F.mse_loss(attn_critical_flat[:, i, :], attn_critical_flat[:, j, :])
    cluster_loss /= (N_crit * (N_crit - 1) / 2)

    ######################################################################### Option with variance (bad!)
    # --- Clustering Loss (Encouraging smooth & compact attention for critical tokens) ---
    # variance = torch.var(attn_critical, dim=-1).mean()  # Variance across spatial dimension
    # cluster_loss = variance  # Lower variance = more compact focus
    ######################################################################### Option with gaussian
    gaussian_loss_critical, gaussian_loss_trivial = gaussian_attention_loss(attn_critical, attn_trivial)

    # --- Separation Loss (Ensuring non-critical tokens avoid critical regions) ---
    avg_critical = attn_critical.mean(dim=1)  # Average across critical tokens
    avg_trivial = attn_trivial.mean(dim=1)  # Average across trivial tokens
    #########################################################################  Option 1
    overlap = (avg_critical * avg_trivial).sum(dim=-1).mean()  # Dot product to measure overlap
    orthog_loss = overlap
    ######################################################################### Option 2
    # attn_critical_norm = F.normalize(attn_critical, p=2, dim=-1)
    # attn_non_critical_norm = F.normalize(attn_trivial, p=2, dim=-1)
    # orthog_loss = (attn_critical_norm @ attn_non_critical_norm.transpose(1, 2)).pow(2).mean()
    #########################################################################
    # --- Coverage Loss (Ensuring all areas are attended to) ---
    combined_attention = avg_critical/2 + avg_trivial/2  # Combine averages
    coverage_loss = F.mse_loss(combined_attention, torch.ones_like(combined_attention) / T)  # Compare to uniform distribution
    
    # --- Final Loss Computation ---
    cluster_scale = 0 #  10e10
    gaussian_scale_critical = 10e4
    gaussian_scale_trivial = 10e-1
    orthog_scale = 10e-1 #  10e-1
    coverage_scale = 10e11

    distance_scale = 0 # 10e-1

    mmd_critical_scale = 0 #1
    mmd_trivial_scale = 0 #1


    #########################################################################  Option to use center of mass
    # 5. Center of Mass Loss: Encourage critical tokens to focus on similar regions (cluster)
    critical_centers_of_mass = compute_center_of_mass(attn_critical, num_patches=256)
    trivial_centers_of_mass = compute_center_of_mass(attn_trivial, num_patches=256)
    distance_loss = center_of_mass_distance_loss(critical_centers_of_mass, trivial_centers_of_mass)
    
    # # Calculate pairwise distances between centers of mass of critical tokens
    centers_of_masses_distance = torch.norm(critical_centers_of_mass[:, :, None, :] - critical_centers_of_mass[:, :, :, None], dim=-1)  # Pairwise distances between centers
    distance_loss = centers_of_masses_distance.mean()  # Encourage smaller distances (i.e., clustering)
    #########################################################################
    if critical_mask is not None:
        #print("critical_mask", critical_mask.shape)
        #print("trivial_mask", trivial_mask.shape)
        critical_mask = critical_mask.view(B, 1, T)
        trivial_mask = trivial_mask.view(B, 1, T)
        #print("split_critical", split_critical.unsqueeze().view(B, 1, T).shape)
        #print("avg_critical", avg_critical.shape)
        mmd_loss_critical = mmd_loss(critical_mask, avg_critical)
        mmd_loss_trivia = mmd_loss(trivial_mask, avg_trivial)
        #print("MMD Loss: critical", mmd_loss(critical_mask, avg_critical))
        #print("MMD Loss: critical", mmd_loss(trivial_mask, avg_trivial))
        #split_loss

    # cluster_scale = 1 / (cluster_loss + 1e-48)
    # #orthog_scale = 1 / (orthog_loss + 1e-8)
    # coverage_scale = 1 / (coverage_loss + 1e-48)
    # distance_scale = 1 / (distance_loss + 1e-48)

#############################################################
    cluster_loss = cluster_scale * cluster_loss
    orthog_loss = orthog_scale * orthog_loss
    coverage_loss = coverage_scale * coverage_loss

    distance_loss = distance_scale * distance_loss

    mmd_loss_critical = mmd_critical_scale * mmd_loss_critical
    mmd_loss_trivia = mmd_trivial_scale * mmd_loss_trivia

    gaussian_loss_critical = gaussian_scale_critical * gaussian_loss_critical
    gaussian_loss_trivial = gaussian_scale_trivial * gaussian_loss_trivial

    total_loss = ( cluster_loss + orthog_loss + coverage_loss + mmd_loss_critical +mmd_loss_trivia + distance_loss + gaussian_loss_critical + gaussian_loss_trivial)

    # #print(f"cluster_loss {cluster_loss}")
    # print(f"gaussian_loss_critical {gaussian_loss_critical}")
    # print(f"gaussian_loss_trivial {gaussian_loss_trivial}")
    # print(f"orthog_loss {orthog_loss}")
    # print(f"coverage_loss {coverage_loss}")
    # # print(f"mmd_loss_critical {mmd_loss_critical}")
    # # print(f"mmd_loss_trivia {mmd_loss_trivia}")
    # print(f"distance_loss {distance_loss}")

    if cluster_loss<0.05 and coverage_loss<0.05:
        print(f"avg_critical {avg_critical[0,:]}")
        print(f"avg_trivial {avg_trivial[0,:]}")

    return total_loss, {
        "cluster_loss": cluster_loss.item(),
        "orthog_loss": orthog_loss.item(),
        "coverage_loss": coverage_loss.item(),
        "mmd_loss_critical": mmd_loss_critical.item(),
        "mmd_loss_trivia": mmd_loss_trivia.item(),

    }

def self_similarity(patch1, patch2):
    """
    Calculate the self-similarity between two patches.
    A simple cosine similarity can be used.
    """
    patch1 = patch1.flatten()
    patch2 = patch2.flatten()
    cos_sim = F.cosine_similarity(patch1.unsqueeze(0), patch2.unsqueeze(0))
    return cos_sim.item()

def pad_patch(patch, patch_size):
    """
    Pad the patch to the desired size (patch_size, patch_size).
    """
    c, h, w = patch.shape
    pad_h = patch_size - h
    pad_w = patch_size - w
    
    # Pad only if necessary
    if pad_h > 0 or pad_w > 0:
        patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    return patch

def expand_shrink_contour(mask, image, contour, patch_size=5, threshold_similarity=0.8):
    """
    Expand or shrink the contour based on patch self-similarity.
    - mask: The binary segmentation mask
    - image: The input image
    - contour: Initial contour points (binary mask)
    - patch_size: Size of the patch to compare around the contour points
    - threshold_similarity: Threshold for self-similarity to determine if the contour should expand or shrink
    """
    # Process each image in the batch individually
    new_contour = torch.zeros_like(mask)

    for b in range(mask.shape[0]):  # Iterate over the batch size
        # Extract the contour for the current batch element (b)
        current_contour = contour[b].squeeze().detach().cpu().numpy()

        # Find contours for this 2D mask
        contours = measure.find_contours(current_contour, 0.5)
        
        for contour_points in contours:
            for point in contour_points:
                y, x = int(point[0]), int(point[1])
                
                # Clip patch coordinates to stay within the image bounds
                y_min = max(y - patch_size // 2, 0)
                y_max = min(y + patch_size // 2 + 1, image.shape[2])
                x_min = max(x - patch_size // 2, 0)
                x_max = min(x + patch_size // 2 + 1, image.shape[3])

                # Extract the patch around the contour point (considering clipping for edges)
                patch = image[b, :, y_min:y_max, x_min:x_max]
                # Pad the patch to the desired patch_size
                patch = pad_patch(patch, patch_size)
                
                # Check for similarity with neighboring patches
                neighbor_patches = []  # List of patches around the contour
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if 0 <= y + dy < image.shape[2] and 0 <= x + dx < image.shape[3]:
                            # Clip the patch coordinates
                            ny_min = max(y + dy - patch_size // 2, 0)
                            ny_max = min(y + dy + patch_size // 2 + 1, image.shape[2])
                            nx_min = max(x + dx - patch_size // 2, 0)
                            nx_max = min(x + dx + patch_size // 2 + 1, image.shape[3])
                            
                            # Extract the neighbor patch
                            neighbor_patch = image[b, :, ny_min:ny_max, nx_min:nx_max]
                            neighbor_patch = pad_patch(neighbor_patch, patch_size)
                            neighbor_patches.append(neighbor_patch)

                similarities = []
                for npatch in neighbor_patches:
                    similarities.append(self_similarity(patch, npatch))

                avg_similarity = np.mean(similarities)

                # If the similarity is above a threshold, expand; otherwise, shrink
                if avg_similarity > threshold_similarity:
                    new_contour[b, 0, y, x] = 1
                else:
                    new_contour[b, 0, y, x] = 0

    return new_contour


def deep_snake(mask, image, iterations=10, patch_size=3):
    """
    Deep snake algorithm for refining a segmentation mask.
    - mask: The initial binary segmentation mask
    - image: The input image
    - iterations: Number of iterations to perform
    - patch_size: Size of the patch to compare around the contour points
    """
    contour = mask.clone()

    for iteration in range(iterations):
        # Expand or shrink the contour based on similarity
        contour = expand_shrink_contour(mask, image, contour, patch_size)
    
    return contour

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

# Superpixel Segmentation with Ordered Labels and Fixed Padding
def segment_superpixels(images, n_segments=81, compactness=10, fixed_superpixels=81):
    """ Segment images into superpixels, return ordered labels, and pad to fixed number. """
    B, C, H, W = images.shape
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (B, H, W, C)
    superpixel_labels = []
    
    for img in images_np:
        img_float = img_as_float(img)
        segments = slic(img_float, n_segments=n_segments, compactness=compactness)
        unique_labels = np.unique(segments)
        centroids = [np.mean(np.argwhere(segments == l), axis=0) for l in unique_labels]
        sorted_indices = np.argsort([c[0] * W + c[1] for c in centroids])
        sorted_labels = unique_labels[sorted_indices]
        
        ordered_segments = np.full((H, W), fixed_superpixels - 1, dtype=np.int64)
        for i, label in enumerate(sorted_labels[:fixed_superpixels]):
            ordered_segments[segments == label] = i
        
        superpixel_labels.append(ordered_segments)
    
    return torch.tensor(np.stack(superpixel_labels)).to("cuda")  # Shape (B, H, W)

# Interpolate Patch Embeddings to Original Resolution
def interpolate_patch_embeddings(vit_output):
    """ Upsample patch embeddings to match original image resolution. """
    B, N, D = vit_output.shape  # (Batch, Num Patches, Patch Dim)
    H, W = int(N ** 0.5), int(N ** 0.5)  # Assuming square patches
    vit_output = vit_output.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
    vit_output = F.interpolate(vit_output, size=(224, 224), mode='bilinear', align_corners=False)
    return vit_output.permute(0, 2, 3, 1)  # (B, 224, 224, D)

# Aggregate Patch Embeddings into Superpixel Representations
def aggregate_superpixel_embeddings(vit_output, superpixel_labels, fixed_superpixels=81):
    """ Aggregate patch embeddings into superpixel embeddings with fixed padding. """
    B, H, W, D = vit_output.shape  # (Batch, 224, 224, Patch Dim)
    superpixel_embeddings = torch.zeros((B, fixed_superpixels, D), device=vit_output.device)
    
    for b in range(B):
        labels = superpixel_labels[b]
        for i in range(fixed_superpixels):
            mask = (labels == i).unsqueeze(-1).expand(-1, -1, D)  # Shape (224, 224, D)
            if mask.any():
                sp_embedding = (vit_output[b] * mask).sum(dim=(0, 1)) / mask.sum()
                superpixel_embeddings[b, i] = sp_embedding
    
    return superpixel_embeddings  # Shape (B, fixed_superpixels, 1024)

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
    
    return hsv_tensor

# Circular Convolution Layer
class CircularConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1, groups=1, padding_mode='circular')
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert (B, fixed_superpixels, 1024) -> (B, 1024, fixed_superpixels)
        x = F.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Convert back

# Node Classification with Soft Mask
class NodeClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)  # Single value for soft mask
    
    def forward(self, h):
        mask = torch.sigmoid(self.fc(h)/ 0.005)  # Low temperature sigmoid
        return mask * h  # Apply soft mask to features

# Define a GNN model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        if data.edge_index.size(1) == 0:
            print("Warning: Empty edge_index")
            return torch.zeros(data.x.size(0), 1024).to(data.x.device)
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class VisionTransformer7x7(nn.Module):
    def __init__(self, image_size=224, patch_size=7, embed_dim=1024, num_heads=16, num_layers=12):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 1024 patches
        self.embed_dim = embed_dim

        # Patch embedding layer (Linear projection of patches)
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, embed_dim)

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

        # Classification Head
        self.head = nn.Linear(embed_dim, 2)  # For ImageNet (change as needed)

    def forward(self, x):
        B, C, H, W = x.shape  # (Batch, 3, 224, 224)

        # Convert image to patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_patches, -1)  # (B, 1024, 147)

        # Project patches into embedding space
        x = self.patch_embed(x)  # (B, 1024, 768)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1025, 768)

        # Add positional embeddings
        x = x + self.pos_embed  # (B, 1025, 768)

        # Transformer Encoder
        x = self.encoder(x)  # (B, 1025, 768)

        # # Classification token output
        # x = self.head(x[:, 1:, :])  # (B, 1000)

        # temperature = 0.05  # Low temperature (small value makes softmax more peaked)
        # scaled_logits = x / temperature
        # #print(f"x shape {x.shape}")
        # softmax_output = F.softmax(scaled_logits, dim=1)
        # #print(softmax_output.shape)
        # x_critical_mask =softmax_output[:,:,0].unsqueeze(1)
        # x_trivial_mask = softmax_output[:,:,1].unsqueeze(1)
        return x

class CLIPWithSoftPrompts(nn.Module):
    def __init__(self, clip_model, prompt_length=10, embedding_dim=512):
        super(CLIPWithSoftPrompts, self).__init__()
        self.clip = clip_model
        self.prompt_length = prompt_length
        self.embedding_dim = embedding_dim
        
        # Initialize soft prompts
        self.soft_prompts = nn.Parameter(torch.randn(prompt_length, embedding_dim))
        
    def forward(self, text_inputs, image_inputs):
        # Encode soft prompts
        soft_prompt_embeddings = self.soft_prompts.unsqueeze(0).expand(text_inputs.size(0), -1, -1)
        
        # Encode text
        text_tokens = self.clip.tokenize(text_inputs).to(self.clip.device)
        text_embeddings = self.clip.encode_text(text_tokens)
        
        # Concatenate soft prompts with text embeddings
        combined_text_embeddings = torch.cat([soft_prompt_embeddings, text_embeddings], dim=1)
        
        # Encode images
        image_embeddings = self.clip.encode_image(image_inputs)
        
        return combined_text_embeddings, image_embeddings

class MultiCLSTokenViT(nn.Module):
    def __init__(self, num_cls_tokens=7):
        super(MultiCLSTokenViT, self).__init__()
        self.num_cls_tokens = num_cls_tokens
        self.config = ViTConfig.from_pretrained('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.vit = ViTModel(self.config)
        
        # Initialize multiple class tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, self.config.hidden_size))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        
        # Adjust the position embeddings to account for the additional class tokens
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        total_tokens = num_patches + num_cls_tokens
        self.position_embeddings = nn.Parameter(
            torch.cat([self.vit.embeddings.position_embeddings[:, :1, :].repeat(1, num_cls_tokens, 1),
                       self.vit.embeddings.position_embeddings[:, 1:num_patches+1, :]], dim=1)
        )
        assert self.position_embeddings.size(1) == total_tokens, "Position embeddings size mismatch"

    def forward(self, pixel_values):
        batch_size = pixel_values.size(0)
        
        # Get the embeddings from the vision transformer
        embeddings = self.vit.embeddings.patch_embeddings(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        
        # Concatenate the class tokens with the embeddings
        cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
        #print(cls_tokens.shape)
        #print(embeddings.shape)
        embeddings = embeddings.transpose(1, 2)
        #print(embeddings.shape)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings.expand(batch_size, -1, -1)
        
        # Pass through the transformer encoder
        encoder_outputs = self.vit.encoder(embeddings)
        #c= self.vit(embeddings).shape
        #print(self.vit)
        #print(f"encoder_outputs {encoder_outputs}")
        
        return encoder_outputs.last_hidden_state

class ExpLICD_Self(nn.Module):  

    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
                #self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
                #self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
            
            config.preprocess = preprocess

            self.model_visual_self_attn = MultiCLSTokenViT(num_cls_tokens=7)
            self.model_visual_self_attn.cuda()
            
            self.model.cuda()
            
            concept_keys = list(concept_list.keys())

            self.concept_token_dict = {}
            for key in concept_keys:
                if config.dataset == 'isic2018':
                    prefix = f"this is a dermoscopic image, the {key} of the lesion is "
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()
        
        self.visual_features = []

        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        layers = [self.model.visual.trunk.blocks[11]]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))
        
        
        self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(7, 768)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
        self.ffn = FFN(768, 768*4)
        self.norm = nn.LayerNorm(768)
        self.proj = nn.Linear(in_features=768, out_features=512, bias=False)

        self.cls_head = nn.Linear(in_features=34, out_features=config.num_class)

        for param in self.model.text.parameters():
            param.requires_grad = False
        for param in self.model.visual.trunk.parameters():
            param.requires_grad = True
        for param in self.model_visual_self_attn.parameters():
            param.requires_grad = True
        
        self.visual_tokens.requires_grad = True
    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.visual_tokens)
        for param in self.cross_attn.parameters():
            param_list.append(param)
        for param in self.ffn.parameters():
            param_list.append(param)
        for param in self.norm.parameters():
            param_list.append(param)
        for param in self.proj.parameters():
            param_list.append(param)
        for param in self.cls_head.parameters():
            param_list.append(param)


        #param_list.append(self.ffn.parameters())
        #param_list.append(self.norm.parameters())
        #param_list.append(self.proj.parameters())
        #param_list.append(self.cls_head.parameters())

        return param_list


    def forward(self, imgs):
        
        self.visual_features.clear()
        #with torch.no_grad():
        #    img_feats, _, _ = self.model(imgs, None)
        #img_feats, _, _ = self.model(imgs, None)
        #print("self visual ", self.visual_features[0][:, 1:, :].shape)
        self_attn_output = self.model_visual_self_attn(imgs)
        cls_tokens = self_attn_output[:, :, :]
        #print(self_attn_output.shape)
        #print("cls_tokens.shape ", cls_tokens.shape)
        # img_feat_map = self.visual_features[0][:, :, :]
        B=cls_tokens.shape[0]
        # B, _, _ = img_feat_map.shape
        visual_tokens = self.visual_tokens.repeat(B, 1, 1)

        # agg_visual_tokens, _ = self.cross_attn(visual_tokens, img_feat_map, img_feat_map)
        #print("agg_visual_tokens ",agg_visual_tokens.shape)
        #agg_visual_tokens = self.proj(self.norm(self.ffn(agg_visual_tokens)))
        agg_visual_tokens, _ = self.cross_attn(visual_tokens, cls_tokens, cls_tokens)
        agg_visual_tokens = self.proj(self.norm(self.ffn(agg_visual_tokens)))
        
        agg_visual_tokens = F.normalize(agg_visual_tokens, dim=-1)
        #print(agg_visual_tokens.shape)

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (self.logit_scale * agg_visual_tokens[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1
        

        image_logits_list = []
        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])
        
        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits) 

        return cls_logits, image_logits_dict



    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.dropout = nn.Dropout(p=0.5)  # Regularization
        self.task = config.dataset
        self.num_of_criteria = NUM_OF_CRITERIA[self.task]
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
            
            config.preprocess = preprocess

            self.model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual
            #print(next(self.model.parameters()).dtype)
            #print(next(self.model_visual_ViT_L.parameters()).dtype)

            # Convert specific layers or parameters to float32
            for param in self.model_visual_ViT_L.parameters():
                param.data = param.data.to(torch.float32)

            self.model_visual_ViT_L.cuda()
            
            self.model.cuda()
            #self.soft_prompt = nn.Embedding(torch.randn(1, 10, 512))
            
            concept_keys = list(concept_list.keys())
            #print("cosine similarity instead of dot products")
            self.concept_token_dict = {}
            for key in concept_keys:
                #if config.dataset == 'ISIC':
                        # if key=="symmetry" or key=="elevation":
                        #     prefix = f"this is a dermoscopic image, the lesion is "
                        # else:
                        #     prefix = f"this is a dermoscopic image, the {key} of the lesion is "
                    #prefix = f"this is a dermoscopic image, the {key} of the lesion is "
                    #prefix = f"this is an ultrasound image of human breasts, the {key} of the tissue is "
                prefix = get_prefix(self.task, key)
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()

            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
        
        self.visual_features = []

        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        layers = [self.model_visual_ViT_L.transformer.resblocks[11]]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))
        
        self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))
        #self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(5, 1024, dtype=torch.float32)))
        #self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(14, 1024, dtype=torch.float32)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.ffn = FFN(1024, 1024*4)
        self.norm = nn.LayerNorm(1024)
        self.proj = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.proj_Sit = nn.Linear(in_features=1024, out_features=768, bias=False)
        self.proj_Sit_to_explicid = nn.Linear(in_features=768, out_features=512, bias=False)
        #                                     34  for ISIC
        #                                     17 for BUSI
        self.cls_head = nn.Linear(in_features=NUM_OF_SIMILARITIES[self.task], out_features=config.num_class)
        #self.cls_head_criteria_only = nn.Linear(in_features=768, out_features=config.num_class)

        # for param in self.model.text.parameters():
        #     param.requires_grad = False
        # for param in self.model.visual.trunk.parameters():
        #     param.requires_grad = False
        for param in self.model_visual_ViT_L.parameters():
            param.requires_grad = True
        
        self.visual_tokens.requires_grad = True
    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.visual_tokens)
        for param in self.cross_attn.parameters():
            param_list.append(param)
        for param in self.ffn.parameters():
            param_list.append(param)
        for param in self.norm.parameters():
            param_list.append(param)
        for param in self.proj.parameters():
            param_list.append(param)
        for param in self.cls_head.parameters():
            param_list.append(param)
        # for param in self.cls_head_criteria_only.parameters():
        #     param_list.append(param)


        return param_list


    def forward(self, imgs, refined_tokens=None):
        
        if refined_tokens is not None:
            #print(f"refined_tokens shape: {refined_tokens.shape}")
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

        self.visual_features.clear()

        _ = self.model_visual_ViT_L(imgs)

        vit_l_output=self.visual_features[0].permute(1, 0, 2)[:, 1:, :] 

        B=vit_l_output.shape[0]

        visual_tokens = self.visual_tokens.repeat(B, 1, 1)
        #visual_tokens=self.dropout(visual_tokens)
        #print(f"vit_l_output.shape {vit_l_output.shape}") # B 256 1024
        agg_visual_tokens, attn_weights = self.cross_attn(visual_tokens, vit_l_output, vit_l_output)
        #print(f"attn_weights shape: {attn_weights.shape}")
        #attn_maps = attn_weights.mean(dim=2)  # Shape: (B, num_tokens, num_tokens)

        # Assuming original image size is (H, W)
        H, W = imgs.shape[2], imgs.shape[3]

        # Extract attention maps for each of the 7 visual tokens
        # for i in range(7):
        #     print(f"attn_maps shape: {attn_maps.shape}")
            #attn_map = attn_maps[:, i, :]  # Shape: (B, num_tokens)
            #attn_map = attn_map.view(B, int(H**0.5), int(W**0.5))  # Reshape to (B, sqrt(num_tokens), sqrt(num_tokens))
            #attn_map = F.interpolate(attn_map.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)  # Resize to (B, H, W)
            
        #print(f"agg_visual_tokens shape: {agg_visual_tokens.shape}") 96 14 1024
        
        agg_visual_tokens_for_explicid = self.proj(self.norm(self.ffn(agg_visual_tokens[:,:self.num_of_criteria,:])))
        agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

        agg_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn(agg_visual_tokens)))

        # agg_visual_tokens_for_explicid = self.proj_Sit_to_explicid(agg_visual_tokens_for_SiT)
        # agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            #print(f"self.concept_token_dict[key] shape: {self.concept_token_dict[key].shape}")
            image_logits_dict[key] = (self.logit_scale * agg_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1
            # # Normalize the embeddings
            # norm_agg_visual_tokens = F.normalize(agg_visual_tokens_for_explicid[:, idx:idx+1, :], dim=-1)
            # norm_concept_embedding = F.normalize(self.concept_token_dict[key], dim=-1)
            
            # # Compute cosine similarity
            # cosine_similarity = torch.matmul(norm_agg_visual_tokens, norm_concept_embedding.repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            
            # image_logits_dict[key] = self.logit_scale * cosine_similarity
            # idx += 1
        
        image_logits_list = []

    # 'asymmetry'
    # 'border irregularity'
    # 'color variation'
    # 'diameter'
    # 'texture'
    # 'vascular patterns'

        img_lgs_asym_bor_col_only = [] # Melanoma
        img_lgs_col_asym_bor_only = [] # Melanocytic Nevus
        img_lgs_vasc_bor_asym_only = [] # Basal Cell Carcinoma
        img_lgs_txtr_vasc_bor_only = [] # Actinic Keratoses
        img_lgs_vasc_txtr_diam_only = [] # Vascular Lesions
        img_lgs_asym_txtr_only = [] # Dermatofibroma
        img_lgs_txtr_vasc_only = [] # Benign Keratosis-like Lesions
        for key in image_logits_dict.keys():
            #print(f"key: {key}")
            #print(f"image_logits_dict[key]: {image_logits_dict[key]}")
            image_logits_list.append(image_logits_dict[key])
            if key == 'asymmetry':
                img_lgs_asym_bor_col_only.append(image_logits_dict[key])
                img_lgs_asym_txtr_only.append(image_logits_dict[key])
                img_lgs_col_asym_bor_only.append(image_logits_dict[key])
                img_lgs_vasc_bor_asym_only.append(image_logits_dict[key])

                img_lgs_txtr_vasc_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_vasc_txtr_diam_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_txtr_vasc_only.append(torch.zeros_like(image_logits_dict[key]))
            elif key == 'border irregularity':
                img_lgs_asym_bor_col_only.append(image_logits_dict[key])
                img_lgs_col_asym_bor_only.append(image_logits_dict[key])
                img_lgs_vasc_bor_asym_only.append(image_logits_dict[key])
                img_lgs_txtr_vasc_bor_only.append(image_logits_dict[key])

                img_lgs_vasc_txtr_diam_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_asym_txtr_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_txtr_vasc_only.append(torch.zeros_like(image_logits_dict[key]))
            elif key == 'color variation':
                img_lgs_asym_bor_col_only.append(image_logits_dict[key])
                img_lgs_col_asym_bor_only.append(image_logits_dict[key])

                img_lgs_txtr_vasc_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_vasc_bor_asym_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_vasc_txtr_diam_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_asym_txtr_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_txtr_vasc_only.append(torch.zeros_like(image_logits_dict[key]))
            elif key == 'diameter':
                img_lgs_vasc_txtr_diam_only.append(image_logits_dict[key])

                img_lgs_asym_bor_col_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_col_asym_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_vasc_bor_asym_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_txtr_vasc_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_asym_txtr_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_txtr_vasc_only.append(torch.zeros_like(image_logits_dict[key]))
            elif key == 'texture':
                img_lgs_txtr_vasc_bor_only.append(image_logits_dict[key])
                img_lgs_asym_txtr_only.append(image_logits_dict[key])
                img_lgs_vasc_txtr_diam_only.append(image_logits_dict[key])
                img_lgs_txtr_vasc_only.append(image_logits_dict[key])

                img_lgs_asym_bor_col_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_col_asym_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_vasc_bor_asym_only.append(torch.zeros_like(image_logits_dict[key]))
            elif key == 'vascular patterns':
                img_lgs_vasc_bor_asym_only.append(image_logits_dict[key])
                img_lgs_vasc_txtr_diam_only.append(image_logits_dict[key])
                img_lgs_txtr_vasc_only.append(image_logits_dict[key])
                img_lgs_txtr_vasc_bor_only.append(image_logits_dict[key])

                img_lgs_asym_bor_col_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_col_asym_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                img_lgs_asym_txtr_only.append(torch.zeros_like(image_logits_dict[key]))

        
        image_logits = torch.cat(image_logits_list, dim=-1)

        img_lgs_AKIEC = torch.cat(img_lgs_txtr_vasc_bor_only, dim=-1)
        img_lgs_BCC = torch.cat(img_lgs_vasc_bor_asym_only, dim=-1)
        img_lgs_BKL = torch.cat(img_lgs_txtr_vasc_only, dim=-1)
        img_lgs_DF = torch.cat(img_lgs_asym_txtr_only, dim=-1)
        img_lgs_MEL = torch.cat(img_lgs_asym_bor_col_only, dim=-1)
        img_lgs_NV = torch.cat(img_lgs_col_asym_bor_only, dim=-1)
        img_lgs_VASC = torch.cat(img_lgs_vasc_txtr_diam_only, dim=-1)

        cls_logits = self.cls_head(image_logits)

        cls_lgs_AKIEC = self.cls_head(img_lgs_AKIEC)
        cls_lgs_BCC = self.cls_head(img_lgs_BCC)
        cls_lgs_BKL = self.cls_head(img_lgs_BKL)
        cls_lgs_DF = self.cls_head(img_lgs_DF)
        cls_lgs_MEL = self.cls_head(img_lgs_MEL)
        cls_lgs_NV = self.cls_head(img_lgs_NV)
        cls_lgs_VASC = self.cls_head(img_lgs_VASC)

        cls_logits_dict = {
            0: cls_lgs_AKIEC,
            1: cls_lgs_BCC,
            2: cls_lgs_BKL,
            3: cls_lgs_DF,
            4: cls_lgs_MEL,
            5: cls_lgs_NV,
            6: cls_lgs_VASC
        }
        #cls_logits_criteria_only = self.cls_head_criteria_only(agg_visual_tokens_for_SiT.mean(dim=1))  

        return cls_logits, None, cls_logits_dict, image_logits_dict,  F.normalize(agg_visual_tokens_for_SiT, dim=-1)

class ExpLICD_ViT_L(nn.Module):  
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.dropout = nn.Dropout(p=0.5)  # Regularization
        self.num_of_criteria = NUM_OF_CRITERIA[config.dataset]
        self.separate_ffs=False
        self.do_logits_similarity=config.do_logits_similarity
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
            
            config.preprocess = preprocess

            self.model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual
            #print(next(self.model.parameters()).dtype)
            #print(next(self.model_visual_ViT_L.parameters()).dtype)

            # Convert specific layers or parameters to float32
            for param in self.model_visual_ViT_L.parameters():
                param.data = param.data.to(torch.float32)

            self.model_visual_ViT_L.cuda()
            
            self.model.cuda()
            #self.soft_prompt = nn.Embedding(torch.randn(1, 10, 512))
            
            concept_keys = list(concept_list.keys())
            #print("cosine similarity instead of dot products")
            self.concept_token_dict = {}
            for key in concept_keys:
                #if config.dataset == 'ISIC':
                        # if key=="symmetry" or key=="elevation":
                        #     prefix = f"this is a dermoscopic image, the lesion is "
                        # else:
                        #     prefix = f"this is a dermoscopic image, the {key} of the lesion is "
                    #prefix = f"this is a dermoscopic image, the {key} of the lesion is "
                    #prefix = f"this is an ultrasound image of human breasts, the {key} of the tissue is "
                prefix = get_prefix(self.config.dataset, key)
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()

            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
        
        self.visual_features = []

        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        layers = [self.model_visual_ViT_L.transformer.resblocks[11]]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))
        
        self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))
        #self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(5, 1024, dtype=torch.float32)))
        #self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(14, 1024, dtype=torch.float32)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.ffn = FFN(1024, 1024*4)
        self.norm = nn.LayerNorm(1024)
        self.proj = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.proj_Sit = nn.Linear(in_features=1024, out_features=768, bias=False)
        self.proj_Sit_to_explicid = nn.Linear(in_features=768, out_features=512, bias=False)

        # Replace single components with ModuleLists
        self.ffns = nn.ModuleList([FFN(1024, 1024*4) for _ in range(self.num_of_criteria)])
        self.norms = nn.ModuleList([nn.LayerNorm(1024) for _ in range(self.num_of_criteria)])
        self.projs = nn.ModuleList([nn.Linear(in_features=1024, out_features=512, bias=False) for _ in range(self.num_of_criteria)])
        self.proj_Sits = nn.ModuleList([nn.Linear(in_features=1024, out_features=768, bias=False) for _ in range(self.num_of_criteria)])
        self.proj_Sits_to_explicids = nn.ModuleList([nn.Linear(in_features=1024, out_features=768, bias=False) for _ in range(self.num_of_criteria)])
        #                                     34  for ISIC
        #                                     17 for BUSI
        self.cls_head = nn.Linear(in_features=NUM_OF_SIMILARITIES[self.config.dataset], out_features=config.num_class)
        #self.cls_head_criteria_only = nn.Linear(in_features=768, out_features=config.num_class)

        # for param in self.model.text.parameters():
        #     param.requires_grad = False
        # for param in self.model.visual.trunk.parameters():
        #     param.requires_grad = False
        for param in self.model_visual_ViT_L.parameters():
            param.requires_grad = True
        
        self.visual_tokens.requires_grad = True
    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.visual_tokens)
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
        # for param in self.cls_head_criteria_only.parameters():
        #     param_list.append(param)


        return param_list


    def forward(self, imgs, refined_tokens=None):
        
        if refined_tokens is not None:
            #print(f"refined_tokens shape: {refined_tokens.shape}")
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

        self.visual_features.clear()

        _ = self.model_visual_ViT_L(imgs)

        vit_l_output=self.visual_features[0].permute(1, 0, 2)[:, 1:, :] 

        B=vit_l_output.shape[0]

        visual_tokens = self.visual_tokens.repeat(B, 1, 1)
        #visual_tokens=self.dropout(visual_tokens)
        #print(f"vit_l_output.shape {vit_l_output.shape}") # B 256 1024
        agg_visual_tokens, attn_weights = self.cross_attn(visual_tokens, vit_l_output, vit_l_output)
        #print(f"attn_weights shape: {attn_weights.shape}")
        #attn_maps = attn_weights.mean(dim=2)  # Shape: (B, num_tokens, num_tokens)

        # Assuming original image size is (H, W)
        H, W = imgs.shape[2], imgs.shape[3]

        # Extract attention maps for each of the 7 visual tokens
        # for i in range(7):
        #     print(f"attn_maps shape: {attn_maps.shape}")
            #attn_map = attn_maps[:, i, :]  # Shape: (B, num_tokens)
            #attn_map = attn_map.view(B, int(H**0.5), int(W**0.5))  # Reshape to (B, sqrt(num_tokens), sqrt(num_tokens))
            #attn_map = F.interpolate(attn_map.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)  # Resize to (B, H, W)
            
        #print(f"agg_visual_tokens shape: {agg_visual_tokens.shape}") 96 14 1024
        if self.separate_ffs:
            processed_tokens_explicid = []
            processed_tokens_sit = []
            for i in range(self.num_of_criteria):
                token = agg_visual_tokens[:,i:i+1,:]  # [B, 1, 1024]
                if i < self.num_of_criteria:
                    processed = self.projs[i](self.norms[i](self.ffns[i](token)))
                    processed_tokens_explicid.append(processed)
                processed_sit = self.proj_Sits[i](self.norms[i](self.ffns[i](token)))
                processed_tokens_sit.append(processed_sit)

            agg_visual_tokens_for_explicid = torch.cat(processed_tokens_explicid, dim=1)
            agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

            agg_visual_tokens_for_SiT = torch.cat(processed_tokens_sit, dim=1)
        else:
            agg_visual_tokens_for_explicid = self.proj(self.norm(self.ffn(agg_visual_tokens[:,:self.num_of_criteria,:])))
            agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)
            
            agg_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn(agg_visual_tokens)))

        # agg_visual_tokens_for_explicid = self.proj_Sit_to_explicid(agg_visual_tokens_for_SiT)
        # agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            #print(f"self.concept_token_dict[key] shape: {self.concept_token_dict[key].shape}")
            image_logits_dict[key] = (self.logit_scale * agg_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1
            # # Normalize the embeddings
            # norm_agg_visual_tokens = F.normalize(agg_visual_tokens_for_explicid[:, idx:idx+1, :], dim=-1)
            # norm_concept_embedding = F.normalize(self.concept_token_dict[key], dim=-1)
            
            # # Compute cosine similarity
            # cosine_similarity = torch.matmul(norm_agg_visual_tokens, norm_concept_embedding.repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            
            # image_logits_dict[key] = self.logit_scale * cosine_similarity
            # idx += 1
        
        image_logits_list = []

    # 'asymmetry'
    # 'border irregularity'
    # 'color variation'
    # 'diameter'
    # 'texture'
    # 'vascular patterns'

        img_lgs_asym_bor_col_only = [] # Melanoma
        img_lgs_col_asym_bor_only = [] # Melanocytic Nevus
        img_lgs_vasc_bor_asym_only = [] # Basal Cell Carcinoma
        img_lgs_txtr_vasc_bor_only = [] # Actinic Keratoses
        img_lgs_vasc_txtr_diam_only = [] # Vascular Lesions
        img_lgs_asym_txtr_only = [] # Dermatofibroma
        img_lgs_txtr_vasc_only = [] # Benign Keratosis-like Lesions
        for key in image_logits_dict.keys():
            #print(f"key: {key}")
            #print(f"image_logits_dict[key] shape: {image_logits_dict[key].shape}")
            image_logits_list.append(image_logits_dict[key])
            if self.do_logits_similarity:
                if key == 'asymmetry':
                    img_lgs_asym_bor_col_only.append(image_logits_dict[key])
                    img_lgs_asym_txtr_only.append(image_logits_dict[key])
                    img_lgs_col_asym_bor_only.append(image_logits_dict[key])
                    img_lgs_vasc_bor_asym_only.append(image_logits_dict[key])

                    img_lgs_txtr_vasc_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_vasc_txtr_diam_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_txtr_vasc_only.append(torch.zeros_like(image_logits_dict[key]))
                elif key == 'border irregularity':
                    img_lgs_asym_bor_col_only.append(image_logits_dict[key])
                    img_lgs_col_asym_bor_only.append(image_logits_dict[key])
                    img_lgs_vasc_bor_asym_only.append(image_logits_dict[key])
                    img_lgs_txtr_vasc_bor_only.append(image_logits_dict[key])

                    img_lgs_vasc_txtr_diam_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_asym_txtr_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_txtr_vasc_only.append(torch.zeros_like(image_logits_dict[key]))
                elif key == 'color variation':
                    img_lgs_asym_bor_col_only.append(image_logits_dict[key])
                    img_lgs_col_asym_bor_only.append(image_logits_dict[key])

                    img_lgs_txtr_vasc_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_vasc_bor_asym_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_vasc_txtr_diam_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_asym_txtr_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_txtr_vasc_only.append(torch.zeros_like(image_logits_dict[key]))
                elif key == 'diameter':
                    img_lgs_vasc_txtr_diam_only.append(image_logits_dict[key])

                    img_lgs_asym_bor_col_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_col_asym_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_vasc_bor_asym_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_txtr_vasc_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_asym_txtr_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_txtr_vasc_only.append(torch.zeros_like(image_logits_dict[key]))
                elif key == 'texture':
                    img_lgs_txtr_vasc_bor_only.append(image_logits_dict[key])
                    img_lgs_asym_txtr_only.append(image_logits_dict[key])
                    img_lgs_vasc_txtr_diam_only.append(image_logits_dict[key])
                    img_lgs_txtr_vasc_only.append(image_logits_dict[key])

                    img_lgs_asym_bor_col_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_col_asym_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_vasc_bor_asym_only.append(torch.zeros_like(image_logits_dict[key]))
                elif key == 'vascular patterns':
                    img_lgs_vasc_bor_asym_only.append(image_logits_dict[key])
                    img_lgs_vasc_txtr_diam_only.append(image_logits_dict[key])
                    img_lgs_txtr_vasc_only.append(image_logits_dict[key])
                    img_lgs_txtr_vasc_bor_only.append(image_logits_dict[key])

                    img_lgs_asym_bor_col_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_col_asym_bor_only.append(torch.zeros_like(image_logits_dict[key]))
                    img_lgs_asym_txtr_only.append(torch.zeros_like(image_logits_dict[key]))

        
        image_logits = torch.cat(image_logits_list, dim=-1)
        #print(f"image_logits shape ", image_logits.shape)

        if self.do_logits_similarity:
            img_lgs_AKIEC = torch.cat(img_lgs_txtr_vasc_bor_only, dim=-1)
            img_lgs_BCC = torch.cat(img_lgs_vasc_bor_asym_only, dim=-1)
            img_lgs_BKL = torch.cat(img_lgs_txtr_vasc_only, dim=-1)
            img_lgs_DF = torch.cat(img_lgs_asym_txtr_only, dim=-1)
            img_lgs_MEL = torch.cat(img_lgs_asym_bor_col_only, dim=-1)
            img_lgs_NV = torch.cat(img_lgs_col_asym_bor_only, dim=-1)
            img_lgs_VASC = torch.cat(img_lgs_vasc_txtr_diam_only, dim=-1)

            cls_lgs_AKIEC = self.cls_head(img_lgs_AKIEC)
            cls_lgs_BCC = self.cls_head(img_lgs_BCC)
            cls_lgs_BKL = self.cls_head(img_lgs_BKL)
            cls_lgs_DF = self.cls_head(img_lgs_DF)
            cls_lgs_MEL = self.cls_head(img_lgs_MEL)
            cls_lgs_NV = self.cls_head(img_lgs_NV)
            cls_lgs_VASC = self.cls_head(img_lgs_VASC)

            cls_logits_dict = {
                0: cls_lgs_AKIEC,
                1: cls_lgs_BCC,
                2: cls_lgs_BKL,
                3: cls_lgs_DF,
                4: cls_lgs_MEL,
                5: cls_lgs_NV,
                6: cls_lgs_VASC
            }
        else:
            cls_logits_dict = {

            }

        cls_logits = self.cls_head(image_logits)


        #cls_logits_criteria_only = self.cls_head_criteria_only(agg_visual_tokens_for_SiT.mean(dim=1))  

        return cls_logits, None, cls_logits_dict, image_logits_dict,  F.normalize(agg_visual_tokens_for_SiT, dim=-1)

class ExpLICD_ViT_L_Multiple_Prompts(nn.Module):  
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.dropout = nn.Dropout(p=0.5)  # Regularization

                # Function to generate multiple prompts
        def generate_prompts(key, concept):
            prefixes = {
                'isic2018': {
                    'symmetry': [
                        f"this is a dermoscopic image, the lesion is {concept}",
                        f"in this skin image, the lesion appears {concept}",
                        f"the melanoma shows {concept} shape",
                        f"examining the dermoscopic image, the lesion exhibits {concept} shape"
                    ],
                    'elevation': [
                        f"this is a dermoscopic image, the lesion is {concept}",
                        f"in this skin image, the lesion appears {concept}",
                        f"the melanoma shows {concept} elevation",
                        f"examining the dermoscopic image, the lesion exhibits {concept} elevation"
                    ],
                    'default': [
                        f"this is a dermoscopic image, the {key} of the lesion is {concept}",
                        f"in this skin image, the {key} appears {concept}",
                        f"the melanoma shows {concept} {key}",
                        f"examining the dermoscopic image, the lesion's {key} is {concept}"
                    ]
                }
            }
            
            if key in ['symmetry', 'elevation']:
                return prefixes['isic2018'][key]
            return prefixes['isic2018']['default']
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
            
            config.preprocess = preprocess

            self.model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual
            #print(next(self.model.parameters()).dtype)
            #print(next(self.model_visual_ViT_L.parameters()).dtype)

            # Convert specific layers or parameters to float32
            for param in self.model_visual_ViT_L.parameters():
                param.data = param.data.to(torch.float32)

            self.model_visual_ViT_L.cuda()
            
            self.model.cuda()
            #self.soft_prompt = nn.Embedding(torch.randn(1, 10, 512))
            
            concept_keys = list(concept_list.keys())

            self.concept_token_dict = {}
            for key in concept_keys:
                if config.dataset == 'isic2018':
                    attr_concept_list = concept_list[key]
                    self.concept_token_dict[key] = []
                    for concept in attr_concept_list:
                        #print(f"key: {key}, concept: {concept}")
                        prompts = generate_prompts(key, concept)
                        tmp_concept_text = self.tokenizer(prompts).cuda()
                        _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                        self.concept_token_dict[key].append(tmp_concept_feats.detach())

            self.logit_scale = logit_scale.detach()

            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
        
        self.visual_features = []

        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        layers = [self.model_visual_ViT_L.transformer.resblocks[11]]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))
        
        
        self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(7, 1024, dtype=torch.float32)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.ffn = FFN(1024, 1024*4)
        self.norm = nn.LayerNorm(1024)
        self.proj = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.proj_Sit = nn.Linear(in_features=1024, out_features=768, bias=False)
        self.proj_Sit_to_explicid = nn.Linear(in_features=768, out_features=512, bias=False)

        self.cls_head = nn.Linear(in_features=34, out_features=config.num_class)

        # for param in self.model.text.parameters():
        #     param.requires_grad = False
        # for param in self.model.visual.trunk.parameters():
        #     param.requires_grad = False
        for param in self.model_visual_ViT_L.parameters():
            param.requires_grad = True
        
        self.visual_tokens.requires_grad = True

        # Method to retrieve a random embedding
    def get_concept_embedding(self, key):
        embeddings_list = self.concept_token_dict[key]
        #print(f"key: {key}")
        #print(f"embeddings_list length: {len(embeddings_list)}")
        #print(f"embeddings_list zero shape: {embeddings_list[0].shape}")
        # Randomly select one 512-sized tensor from each of the 7 tensors
        selected_tensors = [tensor[torch.randint(0, 4, (1,))] for tensor in embeddings_list]
        # Stack the selected tensors to form a tensor of shape (7, 512)
        result_tensor = torch.stack(selected_tensors).squeeze(1)
        #print(f"result_tensor shape: {result_tensor.shape}")
        return embeddings_list

    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.visual_tokens)
        for param in self.cross_attn.parameters():
            param_list.append(param)
        for param in self.ffn.parameters():
            param_list.append(param)
        for param in self.norm.parameters():
            param_list.append(param)
        for param in self.proj.parameters():
            param_list.append(param)
        for param in self.cls_head.parameters():
            param_list.append(param)


        return param_list


    def forward(self, imgs):
        
        self.visual_features.clear()

        _ = self.model_visual_ViT_L(imgs)

        vit_l_output=self.visual_features[0].permute(1, 0, 2)[:, 1:, :] 

        B=vit_l_output.shape[0]

        visual_tokens = self.visual_tokens.repeat(B, 1, 1)
        #visual_tokens=self.dropout(visual_tokens)
        #print(f"vit_l_output.shape {vit_l_output.shape}")
        agg_visual_tokens, _ = self.cross_attn(visual_tokens, vit_l_output, vit_l_output)
        agg_visual_tokens_for_explicid = self.proj(self.norm(self.ffn(agg_visual_tokens)))
        agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

        agg_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn(agg_visual_tokens)))

        # agg_visual_tokens_for_explicid = self.proj_Sit_to_explicid(agg_visual_tokens_for_SiT)
        # agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            concept_embedding = self.get_concept_embedding(key)
            #print(f"agg_visual_tokens_for_explicid[:, idx:idx+1, :]  shape: {agg_visual_tokens_for_explicid[:, idx:idx+1, :] .shape}")
            #print(f"concept_embedding shape: {concept_embedding.shape}")
            
            # image_logits_dict[key] = (self.logit_scale * agg_visual_tokens_for_explicid[:, idx:idx+1, :] @ concept_embedding.repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            # idx += 1
      
            scores = []
    
            for concept_embedding_variation in concept_embedding:
                # Normalize the embeddings
                norm_agg_visual_tokens = F.normalize(agg_visual_tokens_for_explicid[:, idx:idx+1, :], dim=-1)
                
                # Compute dot product for each of the 4 vectors and average the scores
                dot_products = []
                for i in range(concept_embedding_variation.shape[0]):
                    norm_concept_vector = F.normalize(concept_embedding_variation[i], dim=-1)
                    dot_product = torch.matmul(norm_agg_visual_tokens, norm_concept_vector.T).squeeze(1)
                    dot_products.append(dot_product)
                
                avg_dot_product = torch.mean(torch.stack(dot_products), dim=0)
                scores.append(avg_dot_product)
            
            # Average the scores across all concept embeddings
            avg_score = torch.mean(torch.stack(scores), dim=0)
            image_logits_dict[key] = self.logit_scale * avg_score
            idx += 1
        

        image_logits_list = []
        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])
        
        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits) 

        return cls_logits, image_logits_dict,  F.normalize(agg_visual_tokens_for_SiT, dim=-1)

class ExpLICD_ViT_L_Soft_Prompt(nn.Module):  
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
       
        if self.model_name in ['biomedclip', 'openclip']:
               
            self.clip_model, preprocess = clip.load("ViT-L/14", device="cuda")

            # Wrap the CLIP model with soft prompts
            self.model = CLIPWithSoftPrompts(self.clip_model).to("cuda")
            config.preprocess = preprocess

            self.model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual
            #print(next(self.model.parameters()).dtype)
            #print(next(self.model_visual_ViT_L.parameters()).dtype)

            # Convert specific layers or parameters to float32
            for param in self.model_visual_ViT_L.parameters():
                param.data = param.data.to(torch.float32)

            self.model_visual_ViT_L.cuda()
            
            self.model.cuda()
            #self.soft_prompt = nn.Embedding(torch.randn(1, 10, 512))
            
            concept_keys = list(concept_list.keys())

            self.concept_token_dict = {}
            for key in concept_keys:
                if config.dataset == 'isic2018':
                    prefix = f"this is a dermoscopic image, the {key} of the lesion is "
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()

            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
        
        self.visual_features = []

        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        layers = [self.model_visual_ViT_L.transformer.resblocks[11]]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))
        
        
        self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(7, 1024, dtype=torch.float32)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.ffn = FFN(1024, 1024*4)
        self.norm = nn.LayerNorm(1024)
        self.proj = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.proj_Sit = nn.Linear(in_features=1024, out_features=768, bias=False)
        self.proj_Sit_to_explicid = nn.Linear(in_features=768, out_features=512, bias=False)

        self.cls_head = nn.Linear(in_features=34, out_features=config.num_class)

        # for param in self.model.text.parameters():
        #     param.requires_grad = False
        # for param in self.model.visual.trunk.parameters():
        #     param.requires_grad = False
        for param in self.model_visual_ViT_L.parameters():
            param.requires_grad = True
        
        self.visual_tokens.requires_grad = True
    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.visual_tokens)
        for param in self.cross_attn.parameters():
            param_list.append(param)
        for param in self.ffn.parameters():
            param_list.append(param)
        for param in self.norm.parameters():
            param_list.append(param)
        for param in self.proj.parameters():
            param_list.append(param)
        for param in self.cls_head.parameters():
            param_list.append(param)


        return param_list


    def forward(self, imgs):
        
        self.visual_features.clear()

        _ = self.model_visual_ViT_L(imgs)

        vit_l_output=self.visual_features[0].permute(1, 0, 2)[:, 1:, :] 

        B=vit_l_output.shape[0]

        visual_tokens = self.visual_tokens.repeat(B, 1, 1)

        agg_visual_tokens, _ = self.cross_attn(visual_tokens, vit_l_output, vit_l_output)
        agg_visual_tokens_for_explicid = self.proj(self.norm(self.ffn(agg_visual_tokens)))
        agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

        agg_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn(agg_visual_tokens)))

        # agg_visual_tokens_for_explicid = self.proj_Sit_to_explicid(agg_visual_tokens_for_SiT)
        # agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (self.logit_scale * agg_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1
        

        image_logits_list = []
        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])
        
        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits) 

        return cls_logits, image_logits_dict,  F.normalize(agg_visual_tokens_for_SiT, dim=-1)
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
       
        if self.model_name in ['biomedclip', 'openclip']:
               
            self.clip_model, preprocess = clip.load("ViT-L/14", device="cuda")

            # Wrap the CLIP model with soft prompts
            self.model = CLIPWithSoftPrompts(self.clip_model).to("cuda")
            config.preprocess = preprocess

            self.model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual
            #print(next(self.model.parameters()).dtype)
            #print(next(self.model_visual_ViT_L.parameters()).dtype)

            # Convert specific layers or parameters to float32
            for param in self.model_visual_ViT_L.parameters():
                param.data = param.data.to(torch.float32)

            self.model_visual_ViT_L.cuda()
            
            self.model.cuda()
            #self.soft_prompt = nn.Embedding(torch.randn(1, 10, 512))
            
            concept_keys = list(concept_list.keys())

            self.concept_token_dict = {}
            for key in concept_keys:
                if config.dataset == 'isic2018':
                    prefix = f"this is a dermoscopic image, the {key} of the lesion is "
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()

            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
        
        self.visual_features = []

        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        layers = [self.model_visual_ViT_L.transformer.resblocks[11]]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))
        
        
        self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(7, 1024, dtype=torch.float32)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.ffn = FFN(1024, 1024*4)
        self.norm = nn.LayerNorm(1024)
        self.proj = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.proj_Sit = nn.Linear(in_features=1024, out_features=768, bias=False)
        self.proj_Sit_to_explicid = nn.Linear(in_features=768, out_features=512, bias=False)

        self.cls_head = nn.Linear(in_features=34, out_features=config.num_class)

        # for param in self.model.text.parameters():
        #     param.requires_grad = False
        # for param in self.model.visual.trunk.parameters():
        #     param.requires_grad = False
        for param in self.model_visual_ViT_L.parameters():
            param.requires_grad = True
        
        self.visual_tokens.requires_grad = True
    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.visual_tokens)
        for param in self.cross_attn.parameters():
            param_list.append(param)
        for param in self.ffn.parameters():
            param_list.append(param)
        for param in self.norm.parameters():
            param_list.append(param)
        for param in self.proj.parameters():
            param_list.append(param)
        for param in self.cls_head.parameters():
            param_list.append(param)


        return param_list


    def forward(self, imgs):
        
        self.visual_features.clear()

        _ = self.model_visual_ViT_L(imgs)

        vit_l_output=self.visual_features[0].permute(1, 0, 2)[:, 1:, :] 

        B=vit_l_output.shape[0]

        visual_tokens = self.visual_tokens.repeat(B, 1, 1)

        agg_visual_tokens, _ = self.cross_attn(visual_tokens, vit_l_output, vit_l_output)
        agg_visual_tokens_for_explicid = self.proj(self.norm(self.ffn(agg_visual_tokens)))
        agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

        agg_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn(agg_visual_tokens)))

        # agg_visual_tokens_for_explicid = self.proj_Sit_to_explicid(agg_visual_tokens_for_SiT)
        # agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (self.logit_scale * agg_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1
        

        image_logits_list = []
        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])
        
        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits) 

        return cls_logits, image_logits_dict,  F.normalize(agg_visual_tokens_for_SiT, dim=-1)
    

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
class ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens(nn.Module):  
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.num_of_criteria = NUM_OF_CRITERIA[config.dataset]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
            
            config.preprocess = preprocess

            self.model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual
            #self.model_visual_custom = VisionTransformer7x7()

            # Convert specific layers or parameters to float32
            for param in self.model_visual_ViT_L.parameters():
                param.data = param.data.to(torch.float32)

            self.model_visual_ViT_L.cuda()
            #self.model_visual_custom.cuda()
            
            self.model.cuda()
            
            concept_keys = list(concept_list.keys())
            self.concept_token_dict = {}
            for key in concept_keys:
                prefix = get_prefix(self.config.dataset, key)
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()

            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
        
        self.visual_features = []

        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        layers = [self.model_visual_ViT_L.transformer.resblocks[11]]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))
        
        self.critical_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))
        self.trivial_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))

        #self.critical_visual_tokens_pat = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 147, dtype=torch.float32)))
        #self.trivial_visual_tokens_pat = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 147, dtype=torch.float32)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_critical = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_trivial = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_between_patches = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_in_patches = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.ffn = FFN(1024, 1024*4)
        self.norm = nn.LayerNorm(1024)
        self.proj = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.proj_Sit = nn.Linear(in_features=1024, out_features=768, bias=False)
        self.proj_Sit_to_explicid = nn.Linear(in_features=768, out_features=512, bias=False)

        # Replace single components with ModuleLists
        self.ffns = nn.ModuleList([FFN(1024, 1024*4) for _ in range(self.num_of_criteria)])
        self.norms = nn.ModuleList([nn.LayerNorm(1024) for _ in range(self.num_of_criteria)])
        self.projs = nn.ModuleList([nn.Linear(in_features=1024, out_features=512, bias=False) for _ in range(self.num_of_criteria)])
        self.proj_Sits = nn.ModuleList([nn.Linear(in_features=1024, out_features=768, bias=False) for _ in range(self.num_of_criteria)])
        self.proj_Sits_to_explicids = nn.ModuleList([nn.Linear(in_features=1024, out_features=768, bias=False) for _ in range(self.num_of_criteria)])
        #                                     34  for ISIC
        #                                     17 for BUSI
        self.cls_head = nn.Linear(in_features=NUM_OF_SIMILARITIES[self.config.dataset], out_features=config.num_class)

        self.cluster_sigma = torch.nn.Parameter(torch.ones(1))
        self.orthogonal_sigma = torch.nn.Parameter(torch.ones(1))
        self.coverage_sigma = torch.nn.Parameter(torch.ones(1))

        for param in self.model_visual_ViT_L.parameters():
            param.requires_grad = True

        # for param in self.model_visual_custom.parameters():
        #     param.requires_grad = True
        
        self.critical_visual_tokens.requires_grad = True
        self.trivial_visual_tokens.requires_grad = True

        ###########################################################
        self.cnn = PatchSelectorCNN().cuda()
        self.classifying_critical_cnn = ClassifyingCNN().cuda()
        self.classifying_trivial_cnn = ClassifyingCNN().cuda()
        self.gnn = GNN(in_channels=3, hidden_channels=16, out_channels=1024).cuda()

    
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


    def forward(self, imgs, refined_tokens=None, explicid_imgs_latents=None):
        
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

        self.visual_features.clear()

        _ = self.model_visual_ViT_L(imgs)
        #custom_patches = self.model_visual_custom(imgs)
        #print(f"custom_patches shape {custom_patches.shape}")
        patches = image_to_patches(imgs, patch_size=14)
        H_p = W_p = int(patches.shape[1] ** 0.5)
        D_p = patches.shape[2]

        vit_l_output=self.visual_features[0].permute(1, 0, 2)[:, 1:, :] 

        B=vit_l_output.shape[0]

        critical_visual_tokens = self.critical_visual_tokens.repeat(B, 1, 1)
        trivial_visual_tokens = self.trivial_visual_tokens.repeat(B, 1, 1)
        #print(f"vit_l_output.shape in explicd {vit_l_output.shape}") # B 256 1024
        ############################################################################################################# Option 1
        B, T, D = vit_l_output.shape  # B: batch size, T: num_patches (256), D: patch_dim (1024)
        H = W = int(T ** 0.5)  # Assuming T is a perfect square, H = W = 16
        vit_output_unflattened = vit_l_output.view(B, H, W, D).permute(0, 3, 1, 2)  # Shape: (B, 1024, 16, 16)
        #vit_output_unflattened_for_exlusive = vit_output_unflattened.clone().detach()
        #vit_output_unflattened_for_exlusive =vit_output_unflattened
        # critical_mask = self.cnn(vit_output_unflattened)  # Shape: (B, 1, 16, 16)
        # trivial_mask = 1-critical_mask

        # # Multiply the mask with the unflattened vision transformer output
        # crticial_patches = vit_output_unflattened * critical_mask  # Shape: (B, 1024, 16, 16)
        # trivial_patches = vit_output_unflattened * trivial_mask

        # # Flatten the result back to shape (B, 256, 1024)
        # vit_l_output_for_critical = crticial_patches.permute(0, 2, 3, 1).view(B, T, D)  # Shape: (B, 256, 1024)
        # vit_l_output_for_trivial = trivial_patches.permute(0, 2, 3, 1).view(B, T, D)
        # agg_critical_visual_tokens, attn_criticial_weights = self.cross_attn(critical_visual_tokens, vit_l_output_for_critical, vit_l_output_for_critical)
        # agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn(trivial_visual_tokens, vit_l_output_for_trivial, vit_l_output_for_trivial)
        # attention_map_loss, _ = smarter_attention_loss(attn_criticial_weights, attn_trivial_weights)

        #############################################################################################################   Option 2
        # agg_critical_visual_tokens, attn_criticial_weights = self.cross_attn(critical_visual_tokens, vit_l_output, vit_l_output)
        # agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn(trivial_visual_tokens, vit_l_output, vit_l_output)
        # #### This below is good
        # #attention_map_loss, _ = gpt4_0_second_attention_loss(attn_criticial_weights, attn_trivial_weights, self.cluster_sigma, self.orthogonal_sigma, self.coverage_sigma)
        # #print(attention_map_loss.device)
        # attention_map_loss, _ = smarter_attention_loss(attn_criticial_weights, attn_trivial_weights)
        #############################################################################################################  Option 3
        
        critical_mask, trivial_mask = self.cnn(vit_output_unflattened)  # Shape: (B, 1, 16, 16)
        #attention_map_loss=compute_16x16_loss(critical_mask)
        #print(f"compute_16x16_loss loss: {attention_map_loss}")
        overlap_loss=(critical_mask*trivial_mask).sum()
        #print(f"overlap loss: {overlap_loss}")

        #attention_map_loss=overlap_loss/1000
        #print(f"overlap_loss loss: {attention_map_loss}")
        # if attention_map_loss<0.1:
        #     refined_critical_mask = deep_snake(critical_mask, vit_output_unflattened)
        #     crticial_patches = vit_output_unflattened * refined_critical_mask
        #     critical_mask = refined_critical_mask
        ############################################################################################################# Option for cloning vit_output
        # Multiply the mask with the unflattened vision transformer output
        #crticial_patches = vit_output_unflattened * critical_mask  # Shape: (B, 1024, 16, 16)
        #trivial_patches = vit_output_unflattened * trivial_mask

        crticial_patches = vit_output_unflattened  # Shape: (B, 1024, 16, 16)
        trivial_patches = vit_output_unflattened

        cnn_logits_critical = self.classifying_critical_cnn(crticial_patches)
        cnn_logits_trivial = self.classifying_trivial_cnn(trivial_patches)
        # _, attn_between_patches = self.cross_attn_between_patches(
        #     query=vit_l_output,
        #     key=crticial_patches.permute(0, 2, 3, 1).view(B, T, D),
        #     value=crticial_patches.permute(0, 2, 3, 1).view(B, T, D),  
        # )

        # _, attn_in_patches = self.cross_attn_in_patches(
        #     query=crticial_patches.permute(0, 2, 3, 1).view(B, T, D),
        #     key=crticial_patches.permute(0, 2, 3, 1).view(B, T, D),
        #     value=crticial_patches.permute(0, 2, 3, 1).view(B, T, D),  
        # )

        #uniform_attention = torch.full_like(attn_in_patches, 1 / T)*critical_mask.view(B, T, 1)  # Ideal uniform distribution
        #inside_attn_loss = torch.nn.functional.mse_loss(attn_in_patches*critical_mask.view(B, T, 1), uniform_attention)

        #outside_attention = attn_between_patches * (1 - critical_mask.view(B, T, 1))  # Masked attention outside critical areas
        # Penalize high attention values outside critical mask
        #outside_attention_loss = outside_attention.mean()
        #attention_map_loss+=(outside_attention_loss/critical_mask.sum(dim=(-2,-1))).mean()
        #attention_map_loss+=(inside_attn_loss*critical_mask.sum(dim=(-2,-1))).mean()*5
        #attention_map_loss+=critical_mask.sum(dim=(-2,-1)).mean()/10
        #loss_fn = SpatialAwarePatchLoss()
        loss_fn = SpatialAwarePatchAttentionLoss()
       
        #attention_map_loss=loss_fn(vit_output_unflattened, critical_mask)
        #if explicid_imgs_latents is not None:
            #attention_map_loss+=loss_fn(explicid_imgs_latents.view(B, H, W, 768).permute(0, 3, 1, 2), critical_mask)
            #print(f"loss_fn latent loss: {loss_fn(explicid_imgs_latents.view(B, H, W, 768).permute(0, 3, 1, 2), critical_mask)}")
            #attention_map_loss+=loss_fn(vit_output_unflattened, critical_mask)
            #print(f"loss_fn vit_output_unflattened loss: {loss_fn(vit_output_unflattened, critical_mask)}")
            
        #attention_map_loss+=1- critical_mask.sum(dim=(-2,-1)).mean()/256
        #print(f"outside_attention_loss: {outside_attention_loss}")
        #print(f"inside_attn_loss: {inside_attn_loss}")
        #print(f"portion of the trivial {1- critical_mask.sum(dim=(-2,-1)).mean()/256}")
        

        # Flatten the result back to shape (B, 256, 1024)
        vit_l_output_for_critical = crticial_patches.permute(0, 2, 3, 1).view(B, T, D)  # Shape: (B, 256, 1024)
        vit_l_output_for_trivial = trivial_patches.permute(0, 2, 3, 1).view(B, T, D)
        agg_critical_visual_tokens, attn_criticial_weights = self.cross_attn_critical(critical_visual_tokens, vit_l_output_for_critical, vit_l_output_for_critical)
        agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial(trivial_visual_tokens, vit_l_output_for_trivial, vit_l_output_for_trivial)
        #attention_map_loss, _ = gpt4_0_second_attention_loss(attn_criticial_weights, attn_trivial_weights, self.cluster_sigma, self.orthogonal_sigma, self.coverage_sigma)
        attention_map_loss=0
        attention_map_loss += loss_fn(patches.view(B, D_p, H_p, W_p), attn_criticial_weights.view(B, self.num_of_criteria, H, W), attn_trivial_weights.view(B, self.num_of_criteria, H, W))
        # for i in range(7):
        #     #attention_map_loss+=compute_16x16_loss(attn_criticial_weights[:,0,:].unsqueeze(1).view(B, 1, H, W))/7
        #     attention_map_loss += loss_fn(patches.view(B, 588, H, W), attn_criticial_weights[:,0,:].unsqueeze(1).view(B, 1, H, W))
        #print(f"agg_critical_visual_tokens shape {agg_critical_visual_tokens.shape}")
        ############################################################################################################# Option "smarter" attention loss
        #attention_map_loss, _ = smarter_attention_loss(attn_criticial_weights, attn_trivial_weights, critical_mask, trivial_mask)
        #############################################################################################################

        agg_critical_visual_tokens_for_explicid = self.proj(self.norm(self.ffn(agg_critical_visual_tokens[:,:self.num_of_criteria,:])))
        agg_critical_visual_tokens_for_explicid = F.normalize(agg_critical_visual_tokens_for_explicid, dim=-1)
        
        agg_critical_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn(agg_critical_visual_tokens)))

        agg_trivial_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn(agg_trivial_visual_tokens)))

        # agg_visual_tokens_for_explicid = self.proj_Sit_to_explicid(agg_visual_tokens_for_SiT)
        # agg_visual_tokens_for_explicid = F.normalize(agg_visual_tokens_for_explicid, dim=-1)

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (self.logit_scale * agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1

        
        image_logits_list = []

        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])

        
        image_logits = torch.cat(image_logits_list, dim=-1)

        cls_logits_dict = { }

        cls_logits = self.cls_head(image_logits)


        #cls_logits_criteria_only = self.cls_head_criteria_only(agg_visual_tokens_for_SiT.mean(dim=1))
        # print(f"agg_critical_visual_tokens_for_SiT shape: {agg_critical_visual_tokens_for_SiT.shape}")
        # print(f"agg_trivial_visual_tokens_for_SiT shape: {agg_trivial_visual_tokens_for_SiT.shape}")
        #combined_tokens = torch.cat((F.normalize(agg_critical_visual_tokens_for_SiT, dim=-1), F.normalize(agg_trivial_visual_tokens_for_SiT, dim=-1)), dim=1)
        to_return = (patches, cls_logits, None, cls_logits_dict, image_logits_dict,  F.normalize(agg_critical_visual_tokens_for_SiT, dim=-1), 
                     F.normalize(agg_trivial_visual_tokens_for_SiT, dim=-1), attention_map_loss, overlap_loss, attn_criticial_weights, attn_trivial_weights, 
                     vit_l_output, agg_critical_visual_tokens, agg_trivial_visual_tokens, cnn_logits_critical, cnn_logits_trivial, critical_mask, trivial_mask)

        return to_return
    
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
class ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Plus_SuperPixels(nn.Module):  
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.num_of_criteria = NUM_OF_CRITERIA[config.dataset]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
            
            config.preprocess = preprocess

            self.model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual
            self.model_visual_custom = VisionTransformer7x7()

            # Convert specific layers or parameters to float32
            for param in self.model_visual_ViT_L.parameters():
                param.data = param.data.to(torch.float32)

            self.model_visual_ViT_L.cuda()
            self.model_visual_custom.cuda()
            
            self.model.cuda()
            
            concept_keys = list(concept_list.keys())
            self.concept_token_dict = {}
            for key in concept_keys:
                prefix = get_prefix(self.config.dataset, key)
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()

            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
        
        self.visual_features = []

        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        layers = [self.model_visual_ViT_L.transformer.resblocks[11]]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))
        
        self.critical_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))
        self.trivial_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))

        self.critical_visual_tokens_smaller = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))
        self.trivial_visual_tokens_smaller  = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_critical = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_trivial = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)

        self.cross_attn_critical_smaller = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_trivial_smaller = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)

        self.cross_attn_between_patches = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_in_patches = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.ffn = FFN(1024, 1024*4)

        self.ffn_smaller = FFN(1024, 1024*4)
        self.norm_smaller = nn.LayerNorm(1024)

        self.norm = nn.LayerNorm(1024)
        self.proj = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.proj_Sit = nn.Linear(in_features=1024, out_features=768, bias=False)
        self.proj_Sit_to_explicid = nn.Linear(in_features=768, out_features=512, bias=False)

        # Replace single components with ModuleLists
        self.ffns = nn.ModuleList([FFN(1024, 1024*4) for _ in range(self.num_of_criteria)])
        self.norms = nn.ModuleList([nn.LayerNorm(1024) for _ in range(self.num_of_criteria)])
        self.projs = nn.ModuleList([nn.Linear(in_features=1024, out_features=512, bias=False) for _ in range(self.num_of_criteria)])
        self.proj_Sits = nn.ModuleList([nn.Linear(in_features=1024, out_features=768, bias=False) for _ in range(self.num_of_criteria)])
        self.proj_Sits_to_explicids = nn.ModuleList([nn.Linear(in_features=1024, out_features=768, bias=False) for _ in range(self.num_of_criteria)])
        #                                     34  for ISIC
        #                                     17 for BUSI
        self.cls_head = nn.Linear(in_features=NUM_OF_SIMILARITIES[self.config.dataset], out_features=config.num_class)
        self.minimal_cls_head = nn.Linear(in_features=1024*7, out_features=config.num_class)

        self.cluster_sigma = torch.nn.Parameter(torch.ones(1))
        self.orthogonal_sigma = torch.nn.Parameter(torch.ones(1))
        self.coverage_sigma = torch.nn.Parameter(torch.ones(1))

        for param in self.model_visual_ViT_L.parameters():
            param.requires_grad = True

        
        self.critical_visual_tokens.requires_grad = True
        self.trivial_visual_tokens.requires_grad = True

        self.critical_visual_tokens_smaller.requires_grad = True
        self.trivial_visual_tokens_smaller.requires_grad = True

        ###########################################################
        self.cnn = PatchSelectorCNN().cuda()
        self.classifying_critical_cnn = ClassifyingCNN().cuda()
        self.classifying_trivial_cnn = ClassifyingCNN().cuda()
        self.gnn =  CircularConv(in_dim=1024, out_dim=1024).cuda()
        self.node_classifier =  NodeClassifier(in_dim=1024).cuda()

    
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


    def forward(self, imgs, refined_tokens=None, explicid_imgs_latents=None):
        
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

        self.visual_features.clear()
        _ = self.model_visual_ViT_L(imgs)
        patches = image_to_patches(imgs, patch_size=14)
        H_p = W_p = int(patches.shape[1] ** 0.5)
        D_p = patches.shape[2]
        vit_l_output=self.visual_features[0].permute(1, 0, 2)[:, 1:, :] 
        B, T, D = vit_l_output.shape  # B: batch size, T: num_patches (256), D: patch_dim (1024)
        H = W = int(T ** 0.5)  # Assuming T is a perfect square, H = W = 16
        #############################################################################
        #custom_patches = self.model_visual_custom(imgs)[:, 1:, :]
        superpixel_labels = segment_superpixels(imgs)
        vit_output_upsampled = interpolate_patch_embeddings(vit_l_output)
        superpixel_embeddings = aggregate_superpixel_embeddings(vit_output_upsampled, superpixel_labels.to(self.device))
        #T_superpixel = superpixel_embeddings.shape[1]
        #print(f"superpixel_embeddings.shape {superpixel_embeddings.shape}")
        #superpixel_embeddings_unflattened = superpixel_embeddings.view(B, int(T_superpixel ** 0.5), int(T_superpixel ** 0.5), D).permute(0, 3, 1, 2)
        #critical_mask, trivial_mask = self.cnn(superpixel_embeddings_unflattened)
        #crticial_superpixels = (superpixel_embeddings_unflattened * critical_mask).permute(0, 2, 3, 1).view(B, T_superpixel, D)  # Shape: (B, 1024, 16, 16)
        #trivial_superpixels  = (superpixel_embeddings_unflattened * trivial_mask).permute(0, 2, 3, 1).view(B, T_superpixel, D)
        processed_superpixels = self.gnn(superpixel_embeddings)
        crticial_superpixels = self.node_classifier(processed_superpixels)
        #print(f"crticial_superpixels.shape {crticial_superpixels.shape}")
        trivial_superpixels = processed_superpixels - crticial_superpixels
        #print(f"superpixel_embeddings shape {superpixel_embeddings.shape}")

        critical_visual_tokens_smaller = self.critical_visual_tokens_smaller.repeat(B, 1, 1)
        trivial_visual_tokens_smaller = self.trivial_visual_tokens_smaller.repeat(B, 1, 1)

        agg_critical_visual_tokens, attn_critical_weights = self.cross_attn_critical_smaller(critical_visual_tokens_smaller, crticial_superpixels, crticial_superpixels)
        agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial_smaller(trivial_visual_tokens_smaller, trivial_superpixels, trivial_superpixels)
                                                # self.proj(the tokens from X to 512
        agg_critical_visual_tokens_for_explicid = self.proj(self.norm_smaller(self.ffn_smaller(agg_critical_visual_tokens[:,:self.num_of_criteria,:])))
        agg_critical_visual_tokens_for_explicid = F.normalize(agg_critical_visual_tokens_for_explicid, dim=-1)
        #print(f"custom_patches shape {custom_patches.shape}")
        # Generate superpixels for the batch
        #segments_batch = generate_superpixels_batch(imgs)
        #data_list = create_superpixel_graph_batch(imgs, segments_batch)

        #############################################################################
        
        # PLAN make superpixel representation into a square?
        #superpixel_embeddings_unflattened = superpixel_embeddings.view(B, 10, 10, D).permute(0, 3, 1, 2)  # Shape: (B, 1024, 16, 16)
        
        #critical_mask, trivial_mask = self.cnn(vit_output_unflattened)  # Shape: (B, 1, 16, 16)
        critical_mask, trivial_mask = None, None
        overlap_loss = torch.tensor(0.0, device=self.device)
        #overlap_loss=(critical_mask*trivial_mask).sum()

        #crticial_superpixels = vit_output_unflattened * critical_mask  # Shape: (B, 1024, 16, 16)
        #trivial_superpixels  = vit_output_unflattened * trivial_mask

        #loss_fn = SpatialAwarePatchAttentionLoss()
        #vit_l_output_for_critical = crticial_superpixels.permute(0, 2, 3, 1).view(B, T, D)  # Shape: (B, 256, 1024)
        #vit_l_output_for_trivial = trivial_superpixels.permute(0, 2, 3, 1).view(B, T, D)
        attn_explicd_loss=torch.tensor(0.0, device=self.device)
        #attn_explicd_loss += loss_fn(patches.view(B, D_p, H_p, W_p), attn_critical_weights.view(B, self.num_of_criteria, H, W), attn_trivial_weights.view(B, self.num_of_criteria, H, W))
        #############################################################################
        agg_critical_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn(agg_critical_visual_tokens)))
        agg_trivial_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn(agg_trivial_visual_tokens)))

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (self.logit_scale * agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1

        image_logits_list = []

        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])

        image_logits = torch.cat(image_logits_list, dim=-1)
        #cls_logits_dict = { }
        cls_logits = self.cls_head(image_logits)

        # # Compute weighted sum of patches for each token
        # weighted_features = torch.sigmoid(attn_criticial_weights / 0.005).unsqueeze(-1) * vit_l_output_for_critical.unsqueeze(1)  # Shape: (B, 7, 256, patch_dim)
        # token_representations = weighted_features.sum(dim=2)  # Shape: (B, 7, patch_dim)

        # # Concatenate token representations
        # concatenated_tokens = token_representations.view(B, -1)  # Shape: (B, 7 * patch_dim)

        to_return_dict = {
            "patches":patches,
            "critical_mask":critical_mask,
            "trivial_mask":trivial_mask,
            "cls_logits":cls_logits,
            "image_logits_dict":image_logits_dict,
            "agg_critical_visual_tokens_for_SiT":F.normalize(agg_critical_visual_tokens_for_SiT, dim=-1),
            "agg_trivial_visual_tokens_for_SiT":F.normalize(agg_trivial_visual_tokens_for_SiT, dim=-1),
            "attn_explicd_loss":attn_explicd_loss,
            "overlap_loss":overlap_loss,
            "attn_critical_weights":attn_critical_weights,
            "attn_trivial_weights":attn_trivial_weights,
            "vit_l_output":vit_l_output,
            "agg_critical_visual_tokens":agg_critical_visual_tokens,
            "agg_trivial_visual_tokens":agg_trivial_visual_tokens
        }

        doing_cnn_critical=True
        doing_cnn_trivial=True

        if doing_cnn_critical:
            #to_return_dict["cnn_logits_critical"] = self.classifying_critical_cnn(crticial_superpixels)
            to_return_dict["cnn_logits_critical"] = None

        if doing_cnn_trivial:
            #to_return_dict["cnn_logits_trivial"] = self.classifying_trivial_cnn(trivial_superpixels)
            to_return_dict["cnn_logits_trivial"] = None

        #cls_minimal_logits = self.minimal_cls_head(concatenated_tokens)
            
        # to_return = (patches, patches, cls_logits, cls_minimal_logits, cls_logits_dict, image_logits_dict,  F.normalize(agg_critical_visual_tokens_for_SiT, dim=-1), 
        #              F.normalize(agg_trivial_visual_tokens_for_SiT, dim=-1), attention_map_loss, overlap_loss, attn_criticial_weights, attn_trivial_weights, 
        #              vit_l_output, agg_critical_visual_tokens, agg_trivial_visual_tokens, cnn_logits_critical, cnn_logits_trivial, critical_mask, trivial_mask)

        return to_return_dict
    

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################

class ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Plus_Representation_Learning(nn.Module):  

    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.num_of_criteria = NUM_OF_CRITERIA[config.dataset]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
            
            config.preprocess = preprocess

            #self.model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual

            #####
            model_vis_ViT_L = clip.load(f"ViT-L/14")[0].visual
            
            transformer = model_vis_ViT_L.transformer

            # Remove the last residual attention block from the Sequential container
            new_resblocks = nn.Sequential(*list(transformer.resblocks.children())[:-12])

            # Update the new model's transformer with the modified resblocks
            transformer.resblocks = new_resblocks

            # Initialize 7 separate instances of model_visual_ViT_L
            #self.model_visual_ViT_Ls = [clip.load(f"ViT-L/14")[0].visual for _ in range(7)]
            self.model_visual_ViT_Ls = nn.ModuleList([copy.deepcopy(model_vis_ViT_L) for _ in range(self.num_of_criteria)])
            # Convert specific layers or parameters to float32
            for model_visual_ViT_L in self.model_visual_ViT_Ls:
                for param in model_visual_ViT_L.parameters():
                    param.data = param.data.to(torch.float32)
                model_visual_ViT_L.cuda()
            #####

            #self.model_visual_custom = VisionTransformer7x7()

            # Convert specific layers or parameters to float32
            # for param in self.model_visual_ViT_L.parameters():
            #     param.data = param.data.to(torch.float32)

            #self.model_visual_ViT_L.cuda()
            #self.model_visual_custom.cuda()
            
            self.model.cuda()
            
            concept_keys = list(concept_list.keys())
            self.concept_token_dict = {}
            for key in concept_keys:
                prefix = get_prefix(self.config.dataset, key)
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()

            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model
            del model_vis_ViT_L

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
        
        #self.visual_features = []
        #####
        self.visual_features = [[] for _ in range(self.num_of_criteria)]
        #####
        self.hook_list = []

        # def hook_fn(module, input, output):
        #     self.visual_features.append(output) # detach to aboid saving computation graph
        #                                          # might need to remove if finetune the full model
        # layers = [self.model_visual_ViT_L.transformer.resblocks[11]]
        # for layer in layers:
        #     self.hook_list.append(layer.register_forward_hook(hook_fn))

        #####
        def hook_fn(module, input, output, idx):
            self.visual_features[idx].append(output)  # detach to avoid saving computation graph

        layers = [model_visual_ViT_L.transformer.resblocks[11] for model_visual_ViT_L in self.model_visual_ViT_Ls]
        for idx, layer in enumerate(layers):
            self.hook_list.append(layer.register_forward_hook(lambda module, input, output, idx=idx: hook_fn(module, input, output, idx)))
        #####
        
        self.critical_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))
        self.trivial_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))

        self.critical_visual_tokens_smaller = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))
        self.trivial_visual_tokens_smaller  = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, 1024, dtype=torch.float32)))

        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_critical = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attns_criticals = nn.ModuleList([nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True) for _ in range(self.num_of_criteria)])
        self.cross_attn_trivial = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
    
        self.self_attn_critical = Attention(1024, num_heads=16, qkv_bias=True, qk_norm=True)

        self.cross_attn_critical_smaller = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_trivial_smaller = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)

        self.cross_attn_between_patches = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.cross_attn_in_patches = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.ffn = FFN(1024, 1024*4)

        self.ffn_smaller = FFN(1024, 1024*4)
        self.norm_smaller = nn.LayerNorm(1024)

        self.norm = nn.LayerNorm(1024)
        self.proj = nn.Linear(in_features=1024, out_features=512, bias=False)

        dim_of_sit_tokens = 768
        num_heads_sit = 16
        self.proj_tokens_for_Sit = nn.Linear(in_features=1024, out_features=dim_of_sit_tokens, bias=False)
        self.proj_vit_output_for_Sit = nn.Linear(in_features=1024, out_features=dim_of_sit_tokens, bias=False)
        self.norm_vit_output_for_Sit = nn.LayerNorm(dim_of_sit_tokens)
        self.norm_tokens_for_Sit = nn.LayerNorm(dim_of_sit_tokens)
        self.cross_attn_critical_for_Sit = nn.MultiheadAttention(embed_dim=dim_of_sit_tokens, num_heads=num_heads_sit, batch_first=True)

        self.proj_Sit = nn.Linear(in_features=1024, out_features=768, bias=False)
        #self.proj_Sit_to_explicid = nn.Linear(in_features=768, out_features=512, bias=False)

        #                                     34  for ISIC
        #                                     17 for BUSI
        self.cls_head = nn.Linear(in_features=NUM_OF_SIMILARITIES[self.config.dataset], out_features=config.num_class)
        self.minimal_cls_head = nn.Linear(in_features=1024*7, out_features=config.num_class)

        self.cluster_sigma = torch.nn.Parameter(torch.ones(1))
        self.orthogonal_sigma = torch.nn.Parameter(torch.ones(1))
        self.coverage_sigma = torch.nn.Parameter(torch.ones(1))

        # for param in self.model_visual_ViT_L.parameters():
        #     param.requires_grad = True

        #####
        for model_visual_ViT_L in self.model_visual_ViT_Ls:
            for param in model_visual_ViT_L.parameters():
                    param.requires_grad = True
        #####
        
        self.critical_visual_tokens.requires_grad = True
        self.trivial_visual_tokens.requires_grad = True

        self.critical_visual_tokens_smaller.requires_grad = True
        self.trivial_visual_tokens_smaller.requires_grad = True

        ###########################################################
        self.cnn = PatchSelectorCNN().cuda()
        self.classifying_critical_cnn = ClassifyingCNN().cuda()
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


    def forward(self, imgs, refined_tokens=None, explicid_imgs_latents=None):
        
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

        #self.visual_features.clear()
        #_ = self.model_visual_ViT_L(imgs)
        self.visual_features = [[] for _ in range(self.num_of_criteria)]
        # Forward pass through each model_visual_ViT_L instance
        for idx, model_visual_ViT_L in enumerate(self.model_visual_ViT_Ls):
            _ = model_visual_ViT_L(imgs)
        patches = image_to_patches(imgs, patch_size=14)
        H_p = W_p = int(patches.shape[1] ** 0.5)
        D_p = patches.shape[2]
        # Combine the outputs from the 7 instances
        vit_l_outputs = [self.visual_features[idx][0].permute(1, 0, 2)[:, 1:, :] for idx in range(self.num_of_criteria)]
        #vit_l_output = torch.cat(vit_l_outputs, dim=1)

        #vit_l_output=self.visual_features[0].permute(1, 0, 2)[:, 1:, :] 
        B, T, D = vit_l_outputs[0].shape  # B: batch size, T: num_patches (256), D: patch_dim (1024)
        H = W = int(T ** 0.5)  # Assuming T is a perfect square, H = W = 16

        critical_visual_tokens = []
        attn_critical_weights=[]
        for i in range(7):
            #print(f"self.critical_visual_tokens shape {self.critical_visual_tokens.shape}")
            critical_visual_token = self.critical_visual_tokens[i].repeat(B, 1, 1)
            agg_critical_visual_token, attn_critical_weight = self.cross_attns_criticals[i](critical_visual_token, vit_l_outputs[i], vit_l_outputs[i])
            critical_visual_tokens.append(agg_critical_visual_token)
            attn_critical_weights.append(attn_critical_weight)

        agg_critical_visual_tokens = torch.cat(critical_visual_tokens, dim=1)
        attn_critical_weights = torch.cat(attn_critical_weights, dim=1)
        #critical_visual_tokens = self.critical_visual_tokens.repeat(B, 1, 1)
        trivial_visual_tokens = self.trivial_visual_tokens.repeat(B, 1, 1)

        #vit_output_unflattened = vit_l_outputs[0].view(B, H, W, D).permute(0, 3, 1, 2)  # Shape: (B, 1024, 16, 16)
        #critical_mask, trivial_mask = self.cnn(vit_output_unflattened)  # Shape: (B, 1, 16, 16)
        critical_mask, trivial_mask = torch.zeros(B, 1, 16, 16).cuda(), torch.zeros(B, 1, 16, 16).cuda()

        #loss_fn = SpatialAwarePatchAttentionLoss()
        #vit_l_output_for_critical = (vit_output_unflattened * critical_mask).permute(0, 2, 3, 1).view(B, T, D)  # Shape: (B, 256, 1024)
        #vit_l_output_for_trivial = (vit_output_unflattened * trivial_mask).permute(0, 2, 3, 1).view(B, T, D)

        #vit_l_output_for_critical = self.self_attn_critical(vit_l_output_for_critical)

        #agg_critical_visual_tokens, attn_critical_weights = self.cross_attn_critical(critical_visual_tokens, vit_l_output, vit_l_output)
        agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial(trivial_visual_tokens, vit_l_outputs[0], vit_l_outputs[0])
                                                # self.proj(the tokens from X to 512
        agg_critical_visual_tokens_for_explicid = self.proj(self.norm(self.ffn(agg_critical_visual_tokens[:,:self.num_of_criteria,:])))
        agg_critical_visual_tokens_for_explicid = F.normalize(agg_critical_visual_tokens_for_explicid, dim=-1)

        #vit_output_for_Sit = self.norm_vit_output_for_Sit(self.proj_vit_output_for_Sit(vit_l_output))
        attn_explicd_loss = torch.tensor(0.0, device=self.device)
        #critical_visual_tokens_for_representation = F.normalize(self.proj_tokens_for_Sit(agg_critical_visual_tokens), dim=-1)
        #print(attn_critical_weights.shape)
        #print(attn_critical_weights[0,1,:])
        #loss_fn = nn.MSELoss()
        #attn_explicd_loss = loss_fn(attn_critical_weights, attn_critical_weights_representation)
        # if explicid_imgs_latents is not None:
        #     _, attn_critical_weights_representation = self.cross_attn_critical_for_Sit(critical_visual_tokens_for_representation, explicid_imgs_latents, explicid_imgs_latents)
        #     temperature = 0.5  # Adjust temperature to make softmax sharper or smoother
        #     for i in range(self.num_of_criteria):
        #         normalized_attention_map_1 = F.softmax(attn_critical_weights[:,i,:] / temperature, dim=-1)
        #         normalized_attention_map_2 = F.softmax(attn_critical_weights_representation[:,i,:] / temperature, dim=-1)
                #attn_explicd_loss += F.kl_div(F.log_softmax(normalized_attention_map_1, dim=-1), F.softmax(normalized_attention_map_2, dim=-1), reduction='batchmean')
                #attn_explicd_loss += loss_fn(normalized_attention_map_1, normalized_attention_map_2)
            #print(f"attn_explicd_loss {attn_explicd_loss}")
        #_, attn_critical_weights_representation = self.cross_attn_critical_for_Sit(critical_visual_tokens_for_representation, vit_output_for_Sit, vit_output_for_Sit)
        # for i in range(7):
        #     attn_explicd_loss+=compute_16x16_loss(attn_critical_weights[:,i,:].unsqueeze(1).view(B, 1, H, W))/7
        #attn_explicd_loss = 1 - F.cosine_similarity(attn_critical_weights.view(B, -1), attn_critical_weights_representation.view(B, -1), dim=1)
        #attn_explicd_loss = F.kl_div(F.log_softmax(attn_critical_weights, dim=-1), F.softmax(attn_critical_weights_representation, dim=-1), reduction='batchmean')
        #_, attn_trivial_weights_representation = self.cross_attn_trivial(trivial_visual_tokens, vit_output_for_Sit, vit_output_for_Sit)
        #############################################################################
        #custom_patches = self.model_visual_custom(imgs)[:, 1:, :]
        # superpixel_labels = segment_superpixels(imgs)
        # vit_output_upsampled = interpolate_patch_embeddings(vit_l_output)
        # superpixel_embeddings = aggregate_superpixel_embeddings(vit_output_upsampled, superpixel_labels.to(self.device))
        # #T_superpixel = superpixel_embeddings.shape[1]
        # #print(f"superpixel_embeddings.shape {superpixel_embeddings.shape}")
        # #superpixel_embeddings_unflattened = superpixel_embeddings.view(B, int(T_superpixel ** 0.5), int(T_superpixel ** 0.5), D).permute(0, 3, 1, 2)
        # #critical_mask, trivial_mask = self.cnn(superpixel_embeddings_unflattened)
        # #crticial_superpixels = (superpixel_embeddings_unflattened * critical_mask).permute(0, 2, 3, 1).view(B, T_superpixel, D)  # Shape: (B, 1024, 16, 16)
        # #trivial_superpixels  = (superpixel_embeddings_unflattened * trivial_mask).permute(0, 2, 3, 1).view(B, T_superpixel, D)
        # processed_superpixels = self.gnn(superpixel_embeddings)
        # crticial_superpixels = self.node_classifier(processed_superpixels)
        # #print(f"crticial_superpixels.shape {crticial_superpixels.shape}")
        # trivial_superpixels = processed_superpixels - crticial_superpixels
        # #print(f"superpixel_embeddings shape {superpixel_embeddings.shape}")

        # critical_visual_tokens_smaller = self.critical_visual_tokens_smaller.repeat(B, 1, 1)
        # trivial_visual_tokens_smaller = self.trivial_visual_tokens_smaller.repeat(B, 1, 1)

        # agg_critical_visual_tokens, attn_critical_weights = self.cross_attn_critical_smaller(critical_visual_tokens_smaller, crticial_superpixels, crticial_superpixels)
        # agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial_smaller(trivial_visual_tokens_smaller, trivial_superpixels, trivial_superpixels)
        #                                         # self.proj(the tokens from X to 512
        # agg_critical_visual_tokens_for_explicid = self.proj(self.norm_smaller(self.ffn_smaller(agg_critical_visual_tokens[:,:self.num_of_criteria,:])))
        # agg_critical_visual_tokens_for_explicid = F.normalize(agg_critical_visual_tokens_for_explicid, dim=-1)
        #print(f"custom_patches shape {custom_patches.shape}")
        # Generate superpixels for the batch
        #segments_batch = generate_superpixels_batch(imgs)
        #data_list = create_superpixel_graph_batch(imgs, segments_batch)

        #############################################################################
        
        # PLAN make superpixel representation into a square?
        #superpixel_embeddings_unflattened = superpixel_embeddings.view(B, 10, 10, D).permute(0, 3, 1, 2)  # Shape: (B, 1024, 16, 16)
        
        #critical_mask, trivial_mask = None, None
        overlap_loss = torch.tensor(0.0, device=self.device)
        #overlap_loss=(critical_mask*trivial_mask).sum()

        #crticial_superpixels = vit_output_unflattened * critical_mask  # Shape: (B, 1024, 16, 16)
        #trivial_superpixels  = vit_output_unflattened * trivial_mask

        #loss_fn = SpatialAwarePatchAttentionLoss()
        #vit_l_output_for_critical = crticial_superpixels.permute(0, 2, 3, 1).view(B, T, D)  # Shape: (B, 256, 1024)
        #vit_l_output_for_trivial = trivial_superpixels.permute(0, 2, 3, 1).view(B, T, D)
        #attn_explicd_loss=torch.tensor(0.0, device=self.device)
        #attn_explicd_loss += loss_fn(patches.view(B, D_p, H_p, W_p), attn_critical_weights.view(B, self.num_of_criteria, H, W), attn_trivial_weights.view(B, self.num_of_criteria, H, W))
        #############################################################################
        agg_critical_visual_tokens_for_SiT = self.proj_Sit(self.norm(agg_critical_visual_tokens))
        agg_trivial_visual_tokens_for_SiT = self.proj_Sit(self.norm((agg_trivial_visual_tokens)))

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (self.logit_scale * agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1

        image_logits_list = []

        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])

        image_logits = torch.cat(image_logits_list, dim=-1)
        #cls_logits_dict = { }
        cls_logits = self.cls_head(image_logits)

        # # Compute weighted sum of patches for each token
        # weighted_features = torch.sigmoid(attn_criticial_weights / 0.005).unsqueeze(-1) * vit_l_output_for_critical.unsqueeze(1)  # Shape: (B, 7, 256, patch_dim)
        # token_representations = weighted_features.sum(dim=2)  # Shape: (B, 7, patch_dim)

        # # Concatenate token representations
        # concatenated_tokens = token_representations.view(B, -1)  # Shape: (B, 7 * patch_dim)

        to_return_dict = {
            "patches":patches,
            "patches_colored":patches,
            "critical_mask":critical_mask,
            "trivial_mask":trivial_mask,
            "cls_logits":cls_logits,
            "image_logits_dict":image_logits_dict,
            "agg_critical_visual_tokens_for_SiT":F.normalize(agg_critical_visual_tokens_for_SiT, dim=-1),
            "agg_trivial_visual_tokens_for_SiT":F.normalize(agg_trivial_visual_tokens_for_SiT, dim=-1),
            "attn_explicd_loss":attn_explicd_loss,
            "overlap_loss":overlap_loss,
            "attn_critical_weights":attn_critical_weights,
            "attn_trivial_weights":attn_trivial_weights,
            "vit_l_output":vit_l_outputs[0],
            "agg_critical_visual_tokens":agg_critical_visual_tokens,
            "agg_trivial_visual_tokens":agg_trivial_visual_tokens
        }

        doing_cnn_critical = True
        doing_cnn_trivial=False

        to_return_dict["cnn_logits_critical"] = None
        to_return_dict["cnn_logits_trivial"] = None

        if doing_cnn_critical and explicid_imgs_latents is not None:
            to_return_dict["cnn_logits_critical"] = self.classifying_critical_cnn(explicid_imgs_latents.view(B, 16, 16, 768).permute(0, 3, 1, 2))
        else:
            to_return_dict["cnn_logits_critical"] = cls_logits

        if doing_cnn_trivial:
            to_return_dict["cnn_logits_trivial"] = self.classifying_trivial_cnn(vit_l_outputs[0])

        #cls_minimal_logits = self.minimal_cls_head(concatenated_tokens)
            
        # to_return = (patches, patches, cls_logits, cls_minimal_logits, cls_logits_dict, image_logits_dict,  F.normalize(agg_critical_visual_tokens_for_SiT, dim=-1), 
        #              F.normalize(agg_trivial_visual_tokens_for_SiT, dim=-1), attention_map_loss, overlap_loss, attn_criticial_weights, attn_trivial_weights, 
        #              vit_l_output, agg_critical_visual_tokens, agg_trivial_visual_tokens, cnn_logits_critical, cnn_logits_trivial, critical_mask, trivial_mask)

        return to_return_dict
    

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
    

class ExpLICD_ViT_L_Classic(nn.Module):  
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.num_of_criteria = NUM_OF_CRITERIA[config.dataset]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
            
            config.preprocess = preprocess

            self.model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual

            #self.model_visual_custom = VisionTransformer7x7()

            #Convert specific layers or parameters to float32
            for param in self.model_visual_ViT_L.parameters():
                param.data = param.data.to(torch.float32)

            self.model_visual_ViT_L.cuda()
            #self.model_visual_custom.cuda()
            
            self.model.cuda()
            
            concept_keys = list(concept_list.keys())
            self.concept_token_dict = {}
            for key in concept_keys:
                prefix = get_prefix(self.config.dataset, key)
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
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
        self.hook_list = []

        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
        
        ################# Option BioMedclip
        # layers = [self.model.visual.trunk.blocks[11]]

        # for param in self.model.text.parameters():
        #     param.requires_grad = False
        # for param in self.model.visual.trunk.parameters():
        #     param.requires_grad = True
        ################# Option Clip L
        layers = [self.model_visual_ViT_L.transformer.resblocks[23]]
        #################

        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(hook_fn))

        
        hidden_size = 1024
        num_heads = 16
        self.critical_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, hidden_size, dtype=torch.float32)))
        self.trivial_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, hidden_size, dtype=torch.float32)))

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
        self.proj_vit_output_for_Sit = nn.Linear(in_features=hidden_size, out_features=dim_of_sit_tokens, bias=False)
        self.norm_vit_output_for_Sit = nn.LayerNorm(dim_of_sit_tokens)
        self.norm_tokens_for_Sit = nn.LayerNorm(dim_of_sit_tokens)
        self.cross_attn_critical_for_Sit = nn.MultiheadAttention(embed_dim=dim_of_sit_tokens, num_heads=num_heads_sit, batch_first=True)

        self.proj_Sit = nn.Linear(in_features=hidden_size, out_features=768, bias=False)
        #self.proj_Sit_to_explicid = nn.Linear(in_features=768, out_features=512, bias=False)

        #                                     34  for ISIC
        #                                     17 for BUSI
        self.cls_head = nn.Linear(in_features=NUM_OF_SIMILARITIES[self.config.dataset], out_features=config.num_class)
        self.minimal_cls_head = nn.Linear(in_features=hidden_size*7, out_features=config.num_class)

        self.cluster_sigma = torch.nn.Parameter(torch.ones(1))
        self.orthogonal_sigma = torch.nn.Parameter(torch.ones(1))
        self.coverage_sigma = torch.nn.Parameter(torch.ones(1))

        for param in self.model_visual_ViT_L.parameters():
            param.requires_grad = True


        self.critical_visual_tokens.requires_grad = True
        self.trivial_visual_tokens.requires_grad = True

        self.critical_visual_tokens_smaller.requires_grad = True
        self.trivial_visual_tokens_smaller.requires_grad = True

        ###########################################################
        self.cnn = PatchSelectorCNN().cuda()
        self.classifying_critical_cnn = ClassifyingCNN().cuda()
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


    def forward(self, imgs, refined_tokens=None, explicid_imgs_latents=None):
        
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

        self.visual_features.clear()
        patches = image_to_patches(imgs, patch_size=14)
        ################# Option BioMedclip
        # _, _, _ = self.model(imgs, None)
        # vit_l_output = self.visual_features[0][:, 1:, :]
        ################# Option Clip L
        _ = self.model_visual_ViT_L(imgs)
        vit_l_output=self.visual_features[0].permute(1, 0, 2)[:, 1:, :]
        #################
        B, T, D = vit_l_output.shape  # B: batch size, T: num_patches (256), D: patch_dim (1024)
        H = W = int(T ** 0.5)  # Assuming T is a perfect square, H = W = 16
        ############################################################################   Option use Attention Loss for the tokens
        vit_output_unflattened = vit_l_output.view(B, H, W, D).permute(0, 3, 1, 2)  # Shape: (B, 1024, 16, 16)
        #vit_output_unflattened_for_exlusive = vit_output_unflattened.clone().detach()
        #vit_output_unflattened_for_exlusive =vit_output_unflattened
        critical_mask, trivial_mask = self.cnn(vit_output_unflattened)  # Shape: (B, 1, 16, 16)
        #trivial_mask = 1-critical_mask
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

        agg_critical_visual_tokens, attn_critical_weights = self.cross_attn_critical(critical_visual_tokens, vit_l_output_for_critical, vit_l_output)
        agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial(trivial_visual_tokens, vit_l_output_for_trivial, vit_l_output)
        ############################################################################   Option use self-attention for the tokens
        agg_critical_visual_tokens = self.self_attn_critical(agg_critical_visual_tokens)
        #CAUTION The above line is for the case where the self-attention is used for the tokens
                                                # self.proj(the tokens from X to 512
        agg_critical_visual_tokens_for_explicid = self.proj(self.norm(self.ffn(agg_critical_visual_tokens[:,:self.num_of_criteria,:])))
        agg_critical_visual_tokens_for_explicid = F.normalize(agg_critical_visual_tokens_for_explicid, dim=-1)

        #vit_output_for_Sit = self.norm_vit_output_for_Sit(self.proj_vit_output_for_Sit(vit_l_output))
        attn_explicd_loss = torch.tensor(0.0, device=self.device)

        overlap_loss = torch.tensor(0.0, device=self.device)

        #############################################################################
        agg_critical_visual_tokens_for_SiT = self.proj_Sit(self.norm(agg_critical_visual_tokens))
        agg_trivial_visual_tokens_for_SiT = self.proj_Sit(self.norm((agg_trivial_visual_tokens)))

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            ############################### Option dot product
            image_logits_dict[key] = (self.logit_scale *agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            ############################### Option cosine similarity
            #norm_agg_visual_tokens = F.normalize(agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :], dim=-1)
            #norm_concept_embedding = F.normalize(self.concept_token_dict[key], dim=-1)
            #cosine_similarity = torch.matmul(norm_agg_visual_tokens, norm_concept_embedding.repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            #image_logits_dict[key] = self.logit_scale * cosine_similarity
            ###############################
            idx += 1

        image_logits_list = []

        for key in image_logits_dict.keys():
            #image_logits_list.append(F.softmax(image_logits_dict[key], dim=1))
            image_logits_list.append(image_logits_dict[key])

        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits)


        to_return_dict = {
            "patches":patches,
            "patches_colored":patches,
            "critical_mask":critical_mask,
            "trivial_mask":trivial_mask,
            "cls_logits":cls_logits,
            "image_logits_dict":image_logits_dict,
            "agg_critical_visual_tokens_for_SiT":F.normalize(agg_critical_visual_tokens_for_SiT, dim=-1),
            "agg_trivial_visual_tokens_for_SiT":F.normalize(agg_trivial_visual_tokens_for_SiT, dim=-1),
            "attn_explicd_loss":attn_explicd_loss,
            "overlap_loss":overlap_loss,
            "attn_critical_weights":attn_critical_weights,
            "attn_trivial_weights":attn_trivial_weights,
            "vit_l_output":vit_l_output,
            "agg_critical_visual_tokens":agg_critical_visual_tokens,
            "agg_trivial_visual_tokens":agg_trivial_visual_tokens
        }

        doing_cnn_critical = True
        doing_cnn_trivial=False

        to_return_dict["cnn_logits_critical"] = None
        to_return_dict["cnn_logits_trivial"] = None

        if doing_cnn_critical and explicid_imgs_latents is not None:
            to_return_dict["cnn_logits_critical"] = self.classifying_critical_cnn(explicid_imgs_latents.view(B, 16, 16, 768).permute(0, 3, 1, 2))
        else:
            to_return_dict["cnn_logits_critical"] = cls_logits

        if doing_cnn_trivial:
            to_return_dict["cnn_logits_trivial"] = self.classifying_trivial_cnn(vit_l_output)

 
        return to_return_dict
    

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
    
class ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Branching(nn.Module):  

    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.num_of_criteria = NUM_OF_CRITERIA[config.dataset]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Branching")
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
            
            config.preprocess = preprocess

             # Load the shared stem of the vision transformer
            self.shared_stem = clip.load(f"ViT-L/14")[0].visual
            # Convert specific layers or parameters to float32
            for param in self.shared_stem.parameters():
                param.data = param.data.to(torch.float32)
            
            self.shared_stem.cuda()

             # Create 7 separate branches for the remaining layers
            self.branches = nn.ModuleList([copy.deepcopy(self.shared_stem.transformer.resblocks[14:]) for _ in range(self.num_of_criteria)])

            self.model.cuda()
            
            concept_keys = list(concept_list.keys())
            self.concept_token_dict = {}
            for key in concept_keys:
                prefix = get_prefix(self.config.dataset, key)
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()

            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
        
        self.visual_features = [[] for _ in range(7)]
        # self.hook_list = []

        # def hook_fn(module, input, output, idx):
        #     self.visual_features[idx].append(output)  # detach to avoid saving computation graph

        # layers = [branch[-1] for branch in self.branches]
        # for idx, layer in enumerate(layers):
        #     self.hook_list.append(layer.register_forward_hook(lambda module, input, output, idx=idx: hook_fn(module, input, output, idx)))
        
        hidden_size = 1024
        num_heads = 16
        self.critical_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, hidden_size, dtype=torch.float32)))
        self.trivial_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, hidden_size, dtype=torch.float32)))

        self.cross_attns_criticals = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True) for _ in range(self.num_of_criteria)])
        self.ffns = nn.ModuleList([FFN(hidden_size, hidden_size*4) for _ in range(self.num_of_criteria)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.num_of_criteria)])
        self.norms_before = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.num_of_criteria)])
        self.projs = nn.ModuleList([nn.Linear(in_features=hidden_size, out_features=512, bias=False) for _ in range(self.num_of_criteria)])

        self.cross_attn_trivial = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
    
        self.ffn = FFN(hidden_size, hidden_size*4)
        
        self.ffn_smaller = FFN(hidden_size, hidden_size*4)
        self.norm_smaller = nn.LayerNorm(hidden_size)

        self.norm_before = nn.LayerNorm(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(in_features=hidden_size, out_features=512, bias=False)

        dim_of_sit_tokens = 768
        num_heads_sit = 16
        self.proj_tokens_for_Sit = nn.Linear(in_features=hidden_size, out_features=dim_of_sit_tokens, bias=False)
        self.proj_vit_output_for_Sit = nn.Linear(in_features=hidden_size, out_features=dim_of_sit_tokens, bias=False)
        self.norm_vit_output_for_Sit = nn.LayerNorm(dim_of_sit_tokens)
        self.norm_tokens_for_Sit = nn.LayerNorm(dim_of_sit_tokens)
        self.cross_attn_critical_for_Sit = nn.MultiheadAttention(embed_dim=dim_of_sit_tokens, num_heads=num_heads_sit, batch_first=True)
        self.proj_Sit = nn.Linear(in_features=hidden_size, out_features=768, bias=False)

        #                                     34  for ISIC
        #                                     17 for BUSI
        self.cls_head = nn.Linear(in_features=NUM_OF_SIMILARITIES[self.config.dataset], out_features=config.num_class)
        self.minimal_cls_head = nn.Linear(in_features=1024*7, out_features=config.num_class)

        self.cluster_sigma = torch.nn.Parameter(torch.ones(1))
        self.orthogonal_sigma = torch.nn.Parameter(torch.ones(1))
        self.coverage_sigma = torch.nn.Parameter(torch.ones(1))


        #####
        for param in self.shared_stem.parameters():
            param.requires_grad = True

        for branch in self.branches:
            for param in branch.parameters():
                    param.requires_grad = True

        #####
        
        self.critical_visual_tokens.requires_grad = True
        self.trivial_visual_tokens.requires_grad = True

        ###########################################################
        self.cnn = PatchSelectorCNN().cuda()
        self.classifying_critical_cnn = ClassifyingCNN().cuda()

    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.critical_visual_tokens)
        param_list.append(self.trivial_visual_tokens)

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

        for param in self.cls_head.parameters():
            param_list.append(param)

        return param_list


    def forward(self, imgs, explicid_imgs_latents=None):
        self.visual_features = [[] for _ in range(7)]

        # Forward pass through the shared convolutional layer and the first 7 transformer blocks
        x = self.shared_stem.conv1(imgs)  # Convolutional layer
        x = x.flatten(2).transpose(1, 2)  # Flatten and transpose
        x = self.shared_stem.ln_pre(x)  # Layer normalization
        #print(f"x.shape {x.shape}")
        #print(f"self.shared_stem.positional_embedding.shape {self.shared_stem.positional_embedding.shape}")
        x = x + self.shared_stem.positional_embedding[:x.size(1), :]  # Positional embedding
        for i in range(14):
            x = self.shared_stem.transformer.resblocks[i](x)  # First 7 transformer blocks

        # Forward pass through each branch
        for idx, branch in enumerate(self.branches):
            branch_output = x
            for layer in branch:
                branch_output = layer(branch_output)
            self.visual_features[idx].append(branch_output)

        # Combine the outputs from the 7 branches
        vit_l_outputs = [self.visual_features[idx][0] for idx in range(self.num_of_criteria)]

        patches = image_to_patches(imgs, patch_size=14)
        B, _, _ = vit_l_outputs[0].shape  # B: batch size, T: num_patches (256), D: patch_dim (1024)
        #print(f"vit_l_outputs[0].shape {vit_l_outputs[0].shape}")

        attn_critical_weights=[]
        agg_critical_visual_tokens_1024 = []
        agg_critical_visual_tokens = []
        for i in range(7):
            #print(f"self.critical_visual_tokens shape {self.critical_visual_tokens.shape}")
            critical_visual_token = self.critical_visual_tokens[i].repeat(B, 1, 1)
            #print(critical_visual_token.shape)
            agg_critical_visual_token, attn_critical_weight = self.cross_attns_criticals[i](critical_visual_token, vit_l_outputs[i], vit_l_outputs[i])
            agg_critical_visual_tokens_1024.append(agg_critical_visual_token)
            agg_critical_visual_token_512 = self.projs[i](self.norms[i](self.ffns[i](agg_critical_visual_token)))
            agg_critical_visual_token_512 = F.normalize(agg_critical_visual_token_512, dim=-1)
            #print(agg_critical_visual_token_512.shape)
            agg_critical_visual_tokens.append(agg_critical_visual_token_512)
            attn_critical_weights.append(attn_critical_weight)

        agg_critical_visual_tokens_for_explicid = torch.cat(agg_critical_visual_tokens, dim=1)
        #print(agg_critical_visual_tokens_for_explicid.shape)
        attn_critical_weights = torch.cat(attn_critical_weights, dim=1)
        
        trivial_visual_tokens = self.trivial_visual_tokens.repeat(B, 1, 1)
        critical_mask, trivial_mask = torch.zeros(B, 1, 16, 16), torch.zeros(B, 1, 16, 16)
        agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial(trivial_visual_tokens, vit_l_outputs[0], vit_l_outputs[0])
                                                # self.proj(the tokens from X to 512
        attn_explicd_loss = torch.tensor(0.0, device=self.device)

        overlap_loss = torch.tensor(0.0, device=self.device)
 
        agg_critical_visual_tokens_for_SiT = self.proj_Sit(self.norm(torch.cat(agg_critical_visual_tokens_1024, dim=1)))
        agg_trivial_visual_tokens_for_SiT = self.proj_Sit(self.norm((agg_trivial_visual_tokens)))

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (self.logit_scale * agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1

        image_logits_list = []

        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])

        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits)

        to_return_dict = {
            "patches":patches,
            "patches_colored":patches,
            "critical_mask":critical_mask,
            "trivial_mask":trivial_mask,
            "cls_logits":cls_logits,
            "image_logits_dict":image_logits_dict,
            "agg_critical_visual_tokens_for_SiT":F.normalize(agg_critical_visual_tokens_for_SiT, dim=-1),
            "agg_trivial_visual_tokens_for_SiT":F.normalize(agg_trivial_visual_tokens_for_SiT, dim=-1),
            "attn_explicd_loss":attn_explicd_loss,
            "overlap_loss":overlap_loss,
            "attn_critical_weights":attn_critical_weights,
            "attn_trivial_weights":attn_trivial_weights,
            "vit_l_output":vit_l_outputs[0],
            "agg_critical_visual_tokens":torch.cat(agg_critical_visual_tokens_1024, dim=1),
            "agg_trivial_visual_tokens":agg_trivial_visual_tokens
        }

        doing_cnn_critical = True

        to_return_dict["cnn_logits_critical"] = None
        to_return_dict["cnn_logits_trivial"] = None

        if doing_cnn_critical and explicid_imgs_latents is not None:
            to_return_dict["cnn_logits_critical"] = self.classifying_critical_cnn(explicid_imgs_latents.view(B, 16, 16, 768).permute(0, 3, 1, 2))
        else:
            to_return_dict["cnn_logits_critical"] = cls_logits


        return to_return_dict
    

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################



class ExpLICD_ViT_L_with_Attn_Map_and_Additional_Tokens_Cascade(nn.Module):  


    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()
            
        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
        self.num_of_criteria = NUM_OF_CRITERIA[config.dataset]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Branching")
       
        if self.model_name in ['biomedclip', 'openclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
               
            
            config.preprocess = preprocess

             # Load the shared stem of the vision transformer
            self.shared_stem = clip.load(f"ViT-L/14")[0].visual
            # Convert specific layers or parameters to float32
            for param in self.shared_stem.parameters():
                param.data = param.data.to(torch.float32)
            
            self.shared_stem.cuda()

             # Create 7 separate branches for the remaining layers
            self.branches = nn.ModuleList([copy.deepcopy(self.shared_stem.transformer.resblocks[7:]) for _ in range(self.num_of_criteria//7)])

            self.model.cuda()
            
            concept_keys = list(concept_list.keys())
            self.concept_token_dict = {}
            for key in concept_keys:
                prefix = get_prefix(self.config.dataset, key)
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
            self.logit_scale = logit_scale.detach()

            # Step 1: Move model back to CPU
            self.model = self.model.to('cpu')

            # Step 2: Delete the model (optional, for memory release)
            del self.model

            # Step 3: Empty the CUDA memory cache
            torch.cuda.empty_cache()
        
        self.visual_features = [[] for _ in range(7)]
        # self.hook_list = []

        # def hook_fn(module, input, output, idx):
        #     self.visual_features[idx].append(output)  # detach to avoid saving computation graph

        # layers = [branch[-1] for branch in self.branches]
        # for idx, layer in enumerate(layers):
        #     self.hook_list.append(layer.register_forward_hook(lambda module, input, output, idx=idx: hook_fn(module, input, output, idx)))
        
        hidden_size = 1024
        num_heads = 16
        self.critical_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, hidden_size, dtype=torch.float32)))
        self.trivial_visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_of_criteria, hidden_size, dtype=torch.float32)))

        self.cross_attns_criticals = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True) for _ in range(self.num_of_criteria*5)])
        self.ffns = nn.ModuleList([FFN(hidden_size, hidden_size*4) for _ in range(self.num_of_criteria)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.num_of_criteria)])
        self.norms_before = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.num_of_criteria)])
        self.projs = nn.ModuleList([nn.Linear(in_features=hidden_size, out_features=512, bias=False) for _ in range(self.num_of_criteria)])

        self.cross_attn_trivial = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
    
        self.ffn = FFN(hidden_size, hidden_size*4)
        
        self.ffn_smaller = FFN(hidden_size, hidden_size*4)
        self.norm_smaller = nn.LayerNorm(hidden_size)

        self.norm_before = nn.LayerNorm(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(in_features=hidden_size, out_features=512, bias=False)

        dim_of_sit_tokens = 768
        num_heads_sit = 16
        self.proj_tokens_for_Sit = nn.Linear(in_features=hidden_size, out_features=dim_of_sit_tokens, bias=False)
        self.proj_vit_output_for_Sit = nn.Linear(in_features=hidden_size, out_features=dim_of_sit_tokens, bias=False)
        self.norm_vit_output_for_Sit = nn.LayerNorm(dim_of_sit_tokens)
        self.norm_tokens_for_Sit = nn.LayerNorm(dim_of_sit_tokens)
        self.cross_attn_critical_for_Sit = nn.MultiheadAttention(embed_dim=dim_of_sit_tokens, num_heads=num_heads_sit, batch_first=True)
        self.proj_Sit = nn.Linear(in_features=hidden_size, out_features=768, bias=False)

        #                                     34  for ISIC
        #                                     17 for BUSI
        self.cls_head = nn.Linear(in_features=NUM_OF_SIMILARITIES[self.config.dataset], out_features=config.num_class)
        self.minimal_cls_head = nn.Linear(in_features=1024*7, out_features=config.num_class)

        self.cluster_sigma = torch.nn.Parameter(torch.ones(1))
        self.orthogonal_sigma = torch.nn.Parameter(torch.ones(1))
        self.coverage_sigma = torch.nn.Parameter(torch.ones(1))


        #####
        for param in self.shared_stem.parameters():
            param.requires_grad = True

        for branch in self.branches:
            for param in branch.parameters():
                    param.requires_grad = True

        #####
        
        self.critical_visual_tokens.requires_grad = True
        self.trivial_visual_tokens.requires_grad = True

        ###########################################################
        self.cnn = PatchSelectorCNN().cuda()
        self.classifying_critical_cnn = ClassifyingCNN().cuda()

    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.critical_visual_tokens)
        param_list.append(self.trivial_visual_tokens)

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

        for param in self.cls_head.parameters():
            param_list.append(param)

        return param_list


    def forward(self, imgs, explicid_imgs_latents=None):
        self.visual_features = [[] for _ in range(7)]

        # Forward pass through the shared convolutional layer and the first 7 transformer blocks
        x = self.shared_stem.conv1(imgs)  # Convolutional layer
        x = x.flatten(2).transpose(1, 2)  # Flatten and transpose
        x = self.shared_stem.ln_pre(x)  # Layer normalization
        #print(f"x.shape {x.shape}")
        #print(f"self.shared_stem.positional_embedding.shape {self.shared_stem.positional_embedding.shape}")
        x = x + self.shared_stem.positional_embedding[:x.size(1), :]  # Positional embedding
        for i in range(7):
            x = self.shared_stem.transformer.resblocks[i](x)  # First 7 transformer blocks

        B = x.shape[0]
        critical_visual_tokens = self.critical_visual_tokens.repeat(B, 1, 1)
        # Forward pass through each branch
        for idx, branch in enumerate(self.branches):
            branch_output = x
            for l, layer in enumerate(branch):
                branch_output = layer(branch_output)
                critical_visual_tokens, attn_critical_weights = self.cross_attns_criticals[l](critical_visual_tokens, branch_output, branch_output)
            self.visual_features[idx].append(branch_output)

        agg_critical_visual_tokens_for_explicid = self.proj(self.norm(self.ffn(critical_visual_tokens)))
        agg_critical_visual_tokens_for_explicid = F.normalize(agg_critical_visual_tokens_for_explicid, dim=-1)

        # Combine the outputs from the 7 branches
        vit_l_outputs = [self.visual_features[idx][0] for idx in range(self.num_of_criteria//7)]

        patches = image_to_patches(imgs, patch_size=14)
        
        trivial_visual_tokens = self.trivial_visual_tokens.repeat(B, 1, 1)
        critical_mask, trivial_mask = torch.zeros(B, 1, 16, 16), torch.zeros(B, 1, 16, 16)
        agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial(trivial_visual_tokens, vit_l_outputs[0], vit_l_outputs[0])
                                                # self.proj(the tokens from X to 512
        attn_explicd_loss = torch.tensor(0.0, device=self.device)

        overlap_loss = torch.tensor(0.0, device=self.device)
 
        agg_critical_visual_tokens_for_SiT = self.proj_Sit(self.norm(critical_visual_tokens))
        agg_trivial_visual_tokens_for_SiT = self.proj_Sit(self.norm((agg_trivial_visual_tokens)))

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (self.logit_scale * agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            idx += 1

        image_logits_list = []

        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])

        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits)

        to_return_dict = {
            "patches":patches,
            "patches_colored":patches,
            "critical_mask":critical_mask,
            "trivial_mask":trivial_mask,
            "cls_logits":cls_logits,
            "image_logits_dict":image_logits_dict,
            "agg_critical_visual_tokens_for_SiT":F.normalize(agg_critical_visual_tokens_for_SiT, dim=-1),
            "agg_trivial_visual_tokens_for_SiT":F.normalize(agg_trivial_visual_tokens_for_SiT, dim=-1),
            "attn_explicd_loss":attn_explicd_loss,
            "overlap_loss":overlap_loss,
            "attn_critical_weights":attn_critical_weights,
            "attn_trivial_weights":attn_trivial_weights,
            "vit_l_output":vit_l_outputs[0],
            "agg_critical_visual_tokens":critical_visual_tokens,
            "agg_trivial_visual_tokens":agg_trivial_visual_tokens
        }

        doing_cnn_critical = True

        to_return_dict["cnn_logits_critical"] = None
        to_return_dict["cnn_logits_trivial"] = None

        if doing_cnn_critical and explicid_imgs_latents is not None:
            to_return_dict["cnn_logits_critical"] = self.classifying_critical_cnn(explicid_imgs_latents.view(B, 16, 16, 768).permute(0, 3, 1, 2))
        else:
            to_return_dict["cnn_logits_critical"] = cls_logits


        return to_return_dict
    

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
    

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
            self.concept_token_dict = {}
            for key in concept_keys:
                prefix = get_prefix(self.config.dataset, key)
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                #print(f"tmp_concept_text shape: {tmp_concept_text.shape}")
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            
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
        self.cnn = PatchSelectorCNN().cuda()
        self.cnn_kernel_14 = PatchSelectorCNN_kernel_14().cuda()
        self.classifying_critical_cnn = ClassifyingCNN().cuda()
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


    def forward(self, imgs, refined_tokens=None, explicid_imgs_latents=None):
        
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
        imgs_in_hsv = rgb_to_hsv_torch(imgs)
        hsv_patches = imgs_in_hsv.unfold(2, 14, 14).unfold(3, 14, 14)  # (B, 3, 16, 16, 14, 14)
        hsv_patches = hsv_patches.mean(dim=(-1, -2))  # (B, 3, 16, 16)
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
            agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial(trivial_visual_tokens, vit_l_output_hsv, vit_l_output)
        elif self.new_explicd==0:
            agg_critical_visual_tokens, attn_critical_weights = self.cross_attn_critical(critical_visual_tokens, vit_l_output, vit_l_output)
            agg_trivial_visual_tokens, attn_trivial_weights = self.cross_attn_trivial(trivial_visual_tokens, vit_l_output, vit_l_output)
        d_products = torch.matmul(critical_visual_tokens, vit_l_output_hsv.permute(0, 2, 1))/(D**0.5)

          # Compute dot products (before softmax)
        crit_dot = torch.matmul(critical_visual_tokens, vit_l_output_hsv.permute(0, 2, 1))/(D**0.5)  # (7, 256)
        triv_dot = torch.matmul(trivial_visual_tokens, vit_l_output_hsv.permute(0, 2, 1))/(D**0.5)  # (n_t, 256)

        attn_explicd_loss_dict = {}
        if self.config.dataset == 'IDRID':
            attn_explicd_loss_dict=retinal_loss_function(attn_critical_weights, crit_dot, triv_dot, hsv_patches.view(B, 3, T))

        #attn_explicd_loss += 10*gpt4_0_second_attention_loss_mine(attn_critical_weights, attn_trivial_weights)[0]
        #loss_fn = SpatialAwarePatchLoss_mine()
        loss_fn = SpatialAwarePatchLoss_mine_spatial()
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
        #overlap_loss = 10000 * torch.mean(attn_critical_weights * attn_trivial_weights)

        #############################################################################
        agg_critical_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn_smaller(agg_critical_visual_tokens)))
        agg_trivial_visual_tokens_for_SiT = self.proj_Sit(self.norm(self.ffn_smaller(agg_trivial_visual_tokens)))

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            ############################### Option dot product
            image_logits_dict[key] = (self.logit_scale *agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            ############################### Option cosine similarity
            #norm_agg_visual_tokens = F.normalize(agg_critical_visual_tokens_for_explicid[:, idx:idx+1, :], dim=-1)
            #norm_concept_embedding = F.normalize(self.concept_token_dict[key], dim=-1)
            #cosine_similarity = torch.matmul(norm_agg_visual_tokens, norm_concept_embedding.repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
            #image_logits_dict[key] = self.logit_scale * cosine_similarity
            ###############################
            idx += 1

        image_logits_list = []

        for key in image_logits_dict.keys():
            #image_logits_list.append(F.softmax(image_logits_dict[key], dim=1))
            image_logits_list.append(image_logits_dict[key])

        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits)

        cls_with_te_logits = self.cls_wiht_te(image_logits)

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
            "agg_trivial_visual_tokens":agg_trivial_visual_tokens
        }

        doing_cnn_critical = True
        doing_cnn_trivial=False

        to_return_dict["cnn_logits_critical"] = None
        to_return_dict["cnn_logits_trivial"] = None

        if doing_cnn_critical and explicid_imgs_latents is not None:
            to_return_dict["cnn_logits_critical"] = self.classifying_critical_cnn(explicid_imgs_latents.view(B, 16, 16, 768).permute(0, 3, 1, 2))
        else:
            to_return_dict["cnn_logits_critical"] = cls_logits

        if doing_cnn_trivial:
            to_return_dict["cnn_logits_trivial"] = self.classifying_trivial_cnn(vit_l_output)

        to_return_dict["cls_with_te_logits"] = cls_with_te_logits.squeeze(-1)
        to_return_dict["te_loss"] = 0.00001 * te.nn.functional.entropy_logic_loss(self.cls_wiht_te)
        return to_return_dict