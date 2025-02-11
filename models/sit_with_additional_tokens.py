# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.nn import functional as F
from Explicd.model.utils import FFN, FFN_for_SiT

NUM_OF_CRITERIA = {
    'ISIC': 7,
    'ISIC_MINE': 6,

    'IDRID': 5,
    'IDRID_EDEMA': 6,

    'BUSI': 6,
    'BUSI_SOFT': 6
}

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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
    total_attention = attn_critical.sum(dim=1) + attn_non_critical.sum(dim=1)  # (B, 256)
    total_attention = total_attention / total_attention.max(dim=-1, keepdim=True)[0]  # Normalize

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
    cluster_scale = 0.5
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
    

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        #print("===")
        print(num_classes + use_cfg_embedding)
        #print("===")
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        #print("I")
        if (train and use_dropout) or (force_drop_ids is not None):
            #print("II")
            labels = self.token_drop(labels, force_drop_ids)
        #print("III")
        #print(labels)
        embeddings = self.embedding_table(labels)
        #print("IV")
        #print(embeddings)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

# class SiTBlock(nn.Module):
#     """
#     A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, task=None, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         # self.norm_latent_1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         # self.norm_tokens_1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         self.attn = Attention(
#             hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
#             )
#         self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
#         self.task = task
#         self.num_vis_tokens = NUM_OF_CRITERIA[self.task]
        
#         if "fused_attn" in block_kwargs.keys():
#             self.attn.fused_attn = block_kwargs["fused_attn"]
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         # self.norm_latent_2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         # self.norm_tokens_2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(
#             in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
#             )
        
#         # self.mlp_latent = Mlp(
#         #     in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
#         #     )
#         # self.mlp_tokens = Mlp(
#         #     in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
#         #     )
        
#         self.ffn = FFN(hidden_size, hidden_size*4)
        
#         # self.linear_latent_1 = nn.Sequential(
#         #     nn.Linear(hidden_size, hidden_size, bias=True)
#         # )
#         # self.linear_tokens_1 = nn.Sequential(
#         #     nn.Linear(hidden_size, hidden_size, bias=True)
#         # )

#         # self.linear_latent_2 = nn.Sequential(
#         #     nn.Linear(hidden_size, hidden_size, bias=True)
#         # )
#         # self.linear_tokens_2 = nn.Sequential(
#         #     nn.Linear(hidden_size, hidden_size, bias=True)
#         # )

#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )

#         # self.adaLN_modulation_latent = nn.Sequential(
#         #     nn.SiLU(),
#         #     nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         # )

#         # self.adaLN_modulation_tokens = nn.Sequential(
#         #     nn.SiLU(),
#         #     nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         # )


#     def forward(self, x, c, attn_critical_weights, attn_trivial_weights):

#         # # Split the tensor back into the original modalities
#         # T = x.shape[1] - self.num_vis_tokens
#         # latent_tokens = x[:, :T, :]
#         # visual_tokens_crit = x[:, T:T+self.num_vis_tokens, :]
#         # latent_tokens = latent_tokens + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(latent_tokens), shift_mlp, scale_mlp))
#         # visual_tokens_crit = visual_tokens_crit + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(visual_tokens_crit), shift_mlp, scale_mlp))
#         # combined_embeddings = torch.cat([latent_tokens, visual_tokens_crit], dim=1)

#         # shift_latent_1s=[]
#         # scale_latent_1s=[]
#         # gate_latent_1s=[]
#         # shift_latent_2s=[]
#         # scale_latent_2s=[]
#         # gate_latent_2s=[]

#         # shift_tokens_1s=[]
#         # scale_tokens_1s=[]
#         # gate_tokens_1s=[]
#         # shift_tokens_2s=[]
#         # scale_tokens_2s=[]
#         # gate_tokens_2s=[]

#         shift_all_1s=[]
#         scale_all_1s=[]
#         gate_all_1s=[]
#         shift_all_2s=[]
#         scale_all_2s=[]
#         gate_all_2s=[]

#         for i in range(self.num_vis_tokens*3):
#             shift_all_1, scale_all_1, gate_all_1, shift_all_2, scale_all_2, gate_all_2 = (
#                 self.adaLN_modulation(c[:,i,:]).chunk(6, dim=-1)
#             )

#             shift_all_1s.append(shift_all_1)
#             scale_all_1s.append(scale_all_1)
#             gate_all_1s.append(gate_all_1)
#             shift_all_2s.append(shift_all_2)
#             scale_all_2s.append(scale_all_2)
#             gate_all_2s.append(gate_all_2)

#         x_orignal=self.norm1(x)

#         for i in range(self.num_vis_tokens*3):
#             x = modulate(x, shift_all_1s[i], scale_all_1s[i])

#         T = x.shape[1] - self.num_vis_tokens - self.num_vis_tokens*2
#         #print(f"x shape ", x.shape)
#         latent = x[:, :T, :]
#         #print(f"latent shape ", latent.shape)
#         critical_visual_tokens = x[:, T:T+self.num_vis_tokens, :]
#         #print(f"critical_visual_tokens shape ", critical_visual_tokens.shape)
#         trivial_visual_tokens= x[:, T+self.num_vis_tokens:, :]
#         #print(f"trivial_visual_tokensshape ", trivial_visual_tokens.shape)

#         # latent, attn_criticial_weights = self.cross_attn(latent, critical_visual_tokens, critical_visual_tokens)
#         # latent, attn_trivial_weights = self.cross_attn(latent, trivial_visual_tokens, trivial_visual_tokens)

#         # critical_visual_tokens, attn_criticial_weights = self.cross_attn(critical_visual_tokens, latent, latent)
#         # trivial_visual_tokens, attn_trivial_weights = self.cross_attn(trivial_visual_tokens, latent, latent)

#         # Compute average attention weights
#         avg_attn_critical_weights = attn_critical_weights.mean(dim=1)  # (B, 256)
#         avg_attn_trivial_weights = attn_trivial_weights.mean(dim=1)    # (B, 256)

#         # Normalize attention weights
#         avg_attn_critical_weights = F.softmax(avg_attn_critical_weights, dim=-1)
#         avg_attn_trivial_weights = F.softmax(avg_attn_trivial_weights, dim=-1)

#         # Apply cross attention
#         critical_denoised_latent, _ = self.cross_attn(latent, critical_visual_tokens, critical_visual_tokens)
#         trivial_denoised_latent, _ = self.cross_attn(latent, trivial_visual_tokens, trivial_visual_tokens)

#         # Use average attention weights to guide denoising
#         critical_denoised_latent = critical_denoised_latent * avg_attn_critical_weights.unsqueeze(-1)
#         trivial_denoised_latent = trivial_denoised_latent * avg_attn_trivial_weights.unsqueeze(-1)

#         # Combine denoised latents
#         denoised_latent = latent + self.ffn(self.norm2(critical_denoised_latent + trivial_denoised_latent))

#         # Calculate attention loss
#         #attention_map_loss = gpt4_0_second_attention_loss(attn_critical_weights, attn_trivial_weights)
#         attention_map_loss=0

#         # # Cross-attention
#         # denoised_latent_critical, attn_critical_weights = self.cross_attn(latent, critical_visual_tokens, critical_visual_tokens)
#         # denoised_latent_trivial, attn_trivial_weights = self.cross_attn(latent, trivial_visual_tokens, trivial_visual_tokens)

#         # # Combine denoised latents
#         # latent = denoised_latent_critical + denoised_latent_trivial
#         # attention_map_loss = gpt4_0_attention_loss(attn_critical_weights, attn_trivial_weights)

#         #attention_map_loss = sonnet_compute_attention_loss(attn_criticial_weights, attn_trivial_weights)
#         x = torch.cat([denoised_latent, critical_visual_tokens, trivial_visual_tokens], dim=1)

#         for i in range(self.num_vis_tokens*3):
#             x = gate_all_1s[i].unsqueeze(1) * x

#         x=self.norm3(x)

#         for i in range(self.num_vis_tokens*3):
#             x = gate_all_2s[i].unsqueeze(1) *self.mlp(modulate(x, shift_all_2s[i], scale_all_2s[i]))

#         x = x_orignal + x
#         #print(f"combined_embeddings shape ", x.shape)

#         return x, attention_map_loss

class DenoisingCrossAttention(nn.Module):
    def __init__(self, noisy_dim=768, clear_dim=1024, hidden_dim=1024, num_heads=8):
        super(DenoisingCrossAttention, self).__init__()
        
        # Linear projection to align noisy token dimensions to the clear token's dimension
        self.noisy_proj = nn.Linear(noisy_dim, hidden_dim)
        self.clear_proj = nn.Linear(clear_dim, hidden_dim)
        
        # Multihead attention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        # Output projection to map the result back to the original dimension of the noisy token
        self.output_proj = nn.Linear(hidden_dim, noisy_dim)
    
    def forward(self, noisy_tensor, clear_tensor):
        """
        Args:
        noisy_tensor (torch.Tensor): The noisy tensor of shape (B, 256, 768)
        clear_tensor (torch.Tensor): The clear tensor of shape (B, 256, 1024)
        
        Returns:
        torch.Tensor: The denoised tensor of shape (B, 256, 768)
        """
        # Project the noisy tensor and clear tensor to a common embedding space
        Q = self.noisy_proj(noisy_tensor)  # (B, 256, hidden_dim)
        K_V = self.clear_proj(clear_tensor)  # (B, 256, hidden_dim)
        
        # Multihead attention expects input in the format (seq_len, batch_size, embed_dim)
        Q = Q.transpose(0, 1)  # (256, B, hidden_dim)
        K_V = K_V.transpose(0, 1)  # (256, B, hidden_dim)
        
        # Apply multi-head attention: the noisy tensor attends to the clear tensor
        attn_output, _ = self.multihead_attention(Q, K_V, K_V)  # (256, B, hidden_dim)
        
        # Transpose the output back to the original shape
        attn_output = attn_output.transpose(0, 1)  # (B, 256, hidden_dim)
        
        # Project the output back to the original dimension of the noisy tensor
        denoised_tensor = self.output_proj(attn_output)  # (B, 256, 768)
        
        return denoised_tensor

# class SiTBlock(nn.Module):
#     """
#     A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, task=None, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         self.attn = Attention(
#             hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
#             )
#         self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=16, batch_first=True)
#         self.task = task
#         self.num_vis_tokens = NUM_OF_CRITERIA[self.task]
        
#         if "fused_attn" in block_kwargs.keys():
#             self.attn.fused_attn = block_kwargs["fused_attn"]
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(
#             in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
#             )
        
#         self.ffn = FFN(hidden_size, hidden_size*4)

#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )

#         self.denoising_attention = DenoisingCrossAttention(noisy_dim=hidden_size, clear_dim=1024)


#     def forward(self, x, vit_l_output, shorter_visual_tokens, longer_visual_tokens, attn_critical_weights, attn_trivial_weights):

#         # #print(f"vit_l_output shape {vit_l_output.shape}")
#         # critical_visual_tokens = longer_visual_tokens[:, :self.num_vis_tokens, :]
#         # #print(f"critical_visual_tokens shape {critical_visual_tokens.shape}")
#         # trivial_visual_tokens= longer_visual_tokens[:, self.num_vis_tokens:, :]

#         #print(f"trivial_visual_tokens shape {trivial_visual_tokens.shape}")
#         critical_visual_tokens = shorter_visual_tokens[:, :self.num_vis_tokens, :]
#         #print(f"critical_visual_tokens shape {critical_visual_tokens.shape}")
#         trivial_visual_tokens= shorter_visual_tokens[:, self.num_vis_tokens:, :]

#         # Compute average attention weights
#         avg_attn_critical_weights = attn_critical_weights.mean(dim=1)  # (B, 256)
#         avg_attn_trivial_weights = attn_trivial_weights.mean(dim=1)    # (B, 256)

#         # Normalize attention weights
#         avg_attn_critical_weights = F.softmax(avg_attn_critical_weights, dim=-1)
#         avg_attn_trivial_weights = F.softmax(avg_attn_trivial_weights, dim=-1)

#         # Apply cross attention
#         critical_denoised_latent, attn_latent_critical_weights = self.cross_attn(x, critical_visual_tokens, critical_visual_tokens)
#         trivial_denoised_latent, attn_latent_trivial_weights = self.cross_attn(x, trivial_visual_tokens, trivial_visual_tokens)

#         # Use average attention weights to guide denoising
#         critical_denoised_latent = critical_denoised_latent * avg_attn_critical_weights.unsqueeze(-1)
#         trivial_denoised_latent = trivial_denoised_latent * avg_attn_trivial_weights.unsqueeze(-1)

#         # Combine denoised latents
#         x = self.ffn(self.norm2(critical_denoised_latent + trivial_denoised_latent))

#         #x = self.norm1(self.denoising_attention(x, vit_l_output))

#         # Calculate attention loss
#         #attention_map_loss = gpt4_0_second_attention_loss(attn_critical_weights, attn_trivial_weights)
#         #attention_map_loss = gpt4_0_good_from_explicd_attention_loss(attn_latent_critical_weights, attn_latent_trivial_weights)
#         attention_map_loss =0
#         #longer_visual_tokens = torch.cat((critical_visual_tokens, trivial_visual_tokens), dim=1)
#         shorter_visual_tokens =  torch.cat((critical_visual_tokens, trivial_visual_tokens), dim=1)

#         return x, attention_map_loss, vit_l_output, shorter_visual_tokens, longer_visual_tokens

# class SiTBlock(nn.Module):
#     """
#     A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, task=None, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         self.attn = Attention(
#             hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
#             )
#         self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=16, batch_first=True)
#         self.task = task
#         self.num_vis_tokens = NUM_OF_CRITERIA[self.task]
        
#         if "fused_attn" in block_kwargs.keys():
#             self.attn.fused_attn = block_kwargs["fused_attn"]
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(
#             in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
#             )
        
#         self.ffn = FFN(hidden_size, hidden_size*4)

#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )

#         self.denoising_attention = DenoisingCrossAttention(noisy_dim=hidden_size, clear_dim=1024)


#     def forward(self, x, vit_l_output, shorter_visual_tokens, longer_visual_tokens, attn_critical_weights, attn_trivial_weights):

#         # #print(f"vit_l_output shape {vit_l_output.shape}")
#         # critical_visual_tokens = longer_visual_tokens[:, :self.num_vis_tokens, :]
#         # #print(f"critical_visual_tokens shape {critical_visual_tokens.shape}")
#         # trivial_visual_tokens= longer_visual_tokens[:, self.num_vis_tokens:, :]

#         #print(f"trivial_visual_tokens shape {trivial_visual_tokens.shape}")
#         critical_visual_tokens = shorter_visual_tokens[:, :self.num_vis_tokens, :]
#         #print(f"critical_visual_tokens shape {critical_visual_tokens.shape}")
#         trivial_visual_tokens= shorter_visual_tokens[:, self.num_vis_tokens:, :]

#         # Compute average attention weights
#         avg_attn_critical_weights = attn_critical_weights.mean(dim=1)  # (B, 256)
#         avg_attn_trivial_weights = attn_trivial_weights.mean(dim=1)    # (B, 256)

#         # Normalize attention weights
#         avg_attn_critical_weights = F.softmax(avg_attn_critical_weights, dim=-1)
#         avg_attn_trivial_weights = F.softmax(avg_attn_trivial_weights, dim=-1)

#         # Apply cross attention
#         critical_denoised_latent, attn_latent_critical_weights = self.cross_attn(x, critical_visual_tokens, critical_visual_tokens)
#         trivial_denoised_latent, attn_latent_trivial_weights = self.cross_attn(x, trivial_visual_tokens, trivial_visual_tokens)

#         # Use average attention weights to guide denoising
#         critical_denoised_latent = critical_denoised_latent * avg_attn_critical_weights.unsqueeze(-1)
#         trivial_denoised_latent = trivial_denoised_latent * avg_attn_trivial_weights.unsqueeze(-1)

#         # Combine denoised latents
#         x = self.ffn(self.norm2(critical_denoised_latent + trivial_denoised_latent))

#         #x = self.norm1(self.denoising_attention(x, vit_l_output))

#         # Calculate attention loss
#         #attention_map_loss = gpt4_0_second_attention_loss(attn_critical_weights, attn_trivial_weights)
#         #attention_map_loss = gpt4_0_good_from_explicd_attention_loss(attn_latent_critical_weights, attn_latent_trivial_weights)
#         attention_map_loss =0
#         #longer_visual_tokens = torch.cat((critical_visual_tokens, trivial_visual_tokens), dim=1)
#         shorter_visual_tokens =  torch.cat((critical_visual_tokens, trivial_visual_tokens), dim=1)

#         return x, attention_map_loss, vit_l_output, shorter_visual_tokens, longer_visual_tokens
    
class CrossAttentionImageToken(nn.Module):
    def __init__(self, image_channels=3, embed_dim=768, num_heads=16):
        super().__init__()

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(256, 256, bias=True)
        
        
        # Positional embedding for spatial structure
        #self.pos_embedding = nn.Parameter(torch.randn(1, 224*224, embed_dim))


    def forward(self, x, token, attention_map):
        #x = x + self.pos_embedding  # Add positional encoding
        #print(f"x shape {x.shape}") # ([B, 256, 768])
        #print(f"token shape {token.unsqueeze(1).shape}") # [B, 1, 768]
        #print(f"attention_map shape {attention_map.unsqueeze(1).shape}") # [B, 1, 256]
        # Compute cross-attention
#############################################################################################
        # attn_output, _ = self.cross_attention(x, token.unsqueeze(1), token.unsqueeze(1))
        # #print(f"attn_output shape {attn_output.shape}")
        # attn_output = attn_output * self.linear(attention_map.unsqueeze(1)).permute(0, 2, 1)
#############################################################################################33
        
        attn_output, _ = self.cross_attention(x, token, token)
        #print(f"attn_output shape {attn_output.shape}")
        #print(f"attention_map shape {attention_map.shape}")
        #print(f"attention_map before normalization {attention_map}")
        #attention_map = attention_map / attention_map.sum(dim=(-1, -2), keepdim=True)
        #print(f"attention_map shape {attention_map.shape}")
        #print(f"attention_map after normalization {attention_map}")
        #attn_output = attn_output * (attention_map.permute(0, 2, 1).mean(dim=-1).unsqueeze(-1))
        #print(f"attn_output shape {attn_output.shape}")
        attn_map = torch.sigmoid(attention_map.permute(0, 2, 1).mean(dim=-1)).unsqueeze(-1)
        #print(f"attn_map shape {attn_map.shape}")
        attn_output = attn_output * attn_map
        #print(f"attn_output shape {attn_output.shape}")
        
        # Apply attention mask
        #attn_output = attn_output * F.softmax(attention_map.unsqueeze(1), dim=-1).permute(0, 2, 1)  # (B, H*W, 1024)
        x = x + attn_output
        
        #x = x.permute(0, 2, 1).view(B, 1024, H, W)  # Reshape back to (B, 1024, H, W)
        #output = self.decoder(x)  # Decode to image
        return x

# class SiTBlock(nn.Module):
#     """
#     A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, task=None, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         self.attn = Attention(
#             hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
#             )
#         self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=16, batch_first=True)
#         self.another_cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=16, batch_first=True)
#         self.task = task
#         self.num_vis_tokens = NUM_OF_CRITERIA[self.task]
        
#         if "fused_attn" in block_kwargs.keys():
#             self.attn.fused_attn = block_kwargs["fused_attn"]
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(
#             in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
#             )
        
#         self.ffn = FFN(hidden_size, hidden_size*4)

#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )

#         self.denoising_attention = CrossAttentionImageToken()
#         self.denoising_attention_trivial = CrossAttentionImageToken()
#         self.denoising_attention_critical = CrossAttentionImageToken()
#         self.tokens_norms = nn.LayerNorm(hidden_size)

#         self.cluster_sigma = torch.nn.Parameter(torch.ones(1))
#         self.orthogonal_sigma = torch.nn.Parameter(torch.ones(1))
#         self.coverage_sigma = torch.nn.Parameter(torch.ones(1))

#         self.first_attn_map_sigma = torch.nn.Parameter(torch.ones(1))
#         self.kl_critical_sigma = torch.nn.Parameter(torch.ones(1))
#         self.kl_trivial_sigma = torch.nn.Parameter(torch.ones(1))


#     def forward(self, x, visual_tokens, attn_critical_weights, attn_trivial_weights, c):
#         ###########################################   Option 1
#         #x = self.norm1(x)
#         # critical_visual_tokens = visual_tokens[:, :self.num_vis_tokens, :]
#         # trivial_visual_tokens= visual_tokens[:, self.num_vis_tokens:, :]
#         #print(f"attn_critical_weights shape {attn_critical_weights.shape}")
#         #print(f"visual_tokens shape {visual_tokens.shape}")


#         ########################################### Option 2
#         # _, attn_criticial_weights_sit = self.cross_attn(critical_visual_tokens, x, x)
#         # _, attn_trivial_weights_sit = self.cross_attn(trivial_visual_tokens, x, x)
#         # attention_map_loss, _ = gpt4_0_second_attention_loss(attn_criticial_weights_sit, attn_trivial_weights_sit)
#         # x = self.denoising_attention(x, trivial_visual_tokens, attn_trivial_weights)
#         # x = self.denoising_attention(x, critical_visual_tokens, attn_critical_weights)
#         ###########################################  Option 3
#         # x, _ = self.another_cross_attn(x, visual_tokens, visual_tokens)
#         # _, attn_criticial_weights_sit = self.cross_attn(critical_visual_tokens, x, x)
#         # _, attn_trivial_weights_sit = self.cross_attn(trivial_visual_tokens, x, x)
#         # attention_map_loss, _ = gpt4_0_second_attention_loss(attn_criticial_weights_sit, attn_trivial_weights_sit)
#         # #print(f"mse loss {F.mse_loss(attn_criticial_weights_sit, attn_critical_weights)}")
#         # attention_map_loss+=F.mse_loss(attn_criticial_weights_sit, attn_critical_weights)*100   # Add this?
#         # attention_map_loss+=F.mse_loss(attn_trivial_weights_sit, attn_trivial_weights)*100   # Add this?
#         ###########################################  Option 4
#         # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
#         #     self.adaLN_modulation(c).chunk(6, dim=-1)
#         # )
        
#         # x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

#         # T = x.shape[1] - self.num_vis_tokens*3
#         # latent_tokens = x[:, :T, :]
#         # visual_tokens_crit = x[:, T:T+self.num_vis_tokens, :]
#         # visual_tokens_trivial = x[:, T+self.num_vis_tokens:, :]

#         # latent_tokens = latent_tokens + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(latent_tokens), shift_mlp, scale_mlp))
#         # visual_tokens_crit = visual_tokens_crit + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(visual_tokens_crit), shift_mlp, scale_mlp))
#         # visual_tokens_trivial = visual_tokens_trivial + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(visual_tokens_trivial), shift_mlp, scale_mlp))
        
#         # _, attn_criticial_weights_sit = self.cross_attn(critical_visual_tokens, latent_tokens, latent_tokens)
#         # _, attn_trivial_weights_sit = self.cross_attn(trivial_visual_tokens, latent_tokens, latent_tokens)
#         # attention_map_loss, _ = gpt4_0_second_attention_loss(attn_criticial_weights_sit, attn_trivial_weights_sit)
#         # x = torch.cat([latent_tokens, visual_tokens_crit, visual_tokens_trivial], dim=1)
#         #print(f"mse loss {F.mse_loss(attn_criticial_weights_sit, attn_critical_weights)}")
#         #attention_map_loss+=F.mse_loss(attn_criticial_weights_sit, attn_critical_weights)*100   # Add this?
#         #attention_map_loss+=F.mse_loss(attn_trivial_weights_sit, attn_trivial_weights)*100   # Add this?
#         ###########################################  Option 5
#         T = x.shape[1] - self.num_vis_tokens*3
#         latent_tokens = x[:, :T, :]

#         # critical_visual_tokens = x[:, T:T+self.num_vis_tokens, :]
#         # trivial_visual_tokens=  x[:, T+self.num_vis_tokens:, :]

#         critical_visual_tokens = visual_tokens[:, :self.num_vis_tokens, :]
#         trivial_visual_tokens= visual_tokens[:, self.num_vis_tokens:, :]

#         critical_visual_tokens= self.tokens_norms(self.ffn(critical_visual_tokens))
#         trivial_visual_tokens= self.tokens_norms(self.ffn(trivial_visual_tokens))

#         ###########################################   # Option Current 1
#         # x = torch.cat([latent_tokens, critical_visual_tokens, trivial_visual_tokens], dim=1)
#         # x = x + self.attn(self.norm1(x))
#         # latent_tokens = x[:, :T, :]  
#         ########################################### Option Future 1
#         latent_tokens = self.norm1(self.denoising_attention_trivial(latent_tokens, trivial_visual_tokens, attn_trivial_weights))
#         latent_tokens = self.norm2(self.denoising_attention_critical(latent_tokens, critical_visual_tokens, attn_critical_weights))
#         ###########################################

#         #latent_tokens += self.norm2(self.mlp(latent_tokens))

#         # visual_tokens_crit = x[:, T:T+self.num_vis_tokens, :]
#         # visual_tokens_trivial = x[:, T+self.num_vis_tokens:, :]

#         _, attn_criticial_weights_sit = self.cross_attn(critical_visual_tokens, latent_tokens, latent_tokens)
#         _, attn_trivial_weights_sit = self.cross_attn(trivial_visual_tokens, latent_tokens, latent_tokens)
#         #attention_map_loss, _ = gpt4_0_second_attention_loss(attn_criticial_weights_sit, attn_trivial_weights_sit)
#         attention_map_loss, _ = gpt4_0_second_attention_loss(attn_criticial_weights_sit, attn_trivial_weights_sit, self.cluster_sigma, self.orthogonal_sigma, self.coverage_sigma)

#         #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         # attn_critical_weights_normalized = torch.tensor(attn_critical_weights / attn_critical_weights.sum(), dtype=torch.float32)  Option A1
#         # attn_criticial_weights_sit_normalized = torch.tensor( attn_criticial_weights_sit / attn_criticial_weights_sit.sum(), dtype=torch.float32)

#         # attn_trivial_weights_normalized = torch.tensor(attn_trivial_weights / attn_trivial_weights.sum(), dtype=torch.float32)
#         # attn_trivial_weights_sit_normalized = torch.tensor(attn_trivial_weights_sit / attn_trivial_weights_sit.sum(), dtype=torch.float32)

#         # # KL Divergence (Attention Map 1 as P, Attention Map 2 as Q)
#         # eps = 1e-8
#         # kl_loss_critical = F.kl_div((attn_critical_weights_normalized + eps).log(), (attn_criticial_weights_sit_normalized + eps), reduction='batchmean')
#         # kl_loss_trivial = F.kl_div((attn_trivial_weights_normalized + eps).log(), (attn_trivial_weights_sit_normalized + eps), reduction='batchmean')
#         #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Option A2
#         # Ensure the attention maps are normalized
#         attn_critical_weights = attn_critical_weights / attn_critical_weights.sum(dim=(-1, -2), keepdim=True)
#         attn_criticial_weights_sit = attn_criticial_weights_sit / attn_criticial_weights_sit.sum(dim=(-1, -2), keepdim=True)

#         attn_trivial_weights = attn_trivial_weights / attn_trivial_weights.sum(dim=(-1, -2), keepdim=True)
#         attn_trivial_weights_sit = attn_trivial_weights_sit / attn_trivial_weights_sit.sum(dim=(-1, -2), keepdim=True)
#         # Add a small epsilon to avoid log(0)
#         epsilon = 1e-10
#         attn_critical_weights = attn_critical_weights + epsilon
#         attn_criticial_weights_sit = attn_criticial_weights_sit + epsilon
#         attn_trivial_weights = attn_trivial_weights + epsilon
#         attn_trivial_weights_sit = attn_trivial_weights_sit + epsilon
        
#         # Compute KL divergence
#         kl_loss_critical = F.kl_div(attn_critical_weights.log(), attn_criticial_weights_sit, reduction='batchmean')  # KL(A1 || A2)
#         kl_loss_trivial = F.kl_div(attn_trivial_weights.log(), attn_trivial_weights_sit, reduction='batchmean')  # KL(A1 || A2)

#         first_attn_map_sigma = F.softplus(self.first_attn_map_sigma) + 1e-6
#         kl_critical_sigma = F.softplus(self.kl_critical_sigma) + 1e-6
#         kl_trivial_sigma = F.softplus(self.kl_trivial_sigma) + 1e-6
#         log_weight = 0.1
#         total_attention_map_loss =  (
#         1.5 * (attention_map_loss / (2 * first_attn_map_sigma**2)) +
#         1.0 * (kl_loss_critical / (2 * kl_critical_sigma**2)) +
#         1.5 * (kl_loss_trivial / (2 * kl_trivial_sigma**2)) +
#         log_weight * (torch.log(first_attn_map_sigma) + torch.log(kl_critical_sigma) + torch.log(kl_trivial_sigma))
#     )
#         #print(f"total_attention_map_loss {total_attention_map_loss}")
#         #attention_map_loss+=kl_loss_critical + kl_loss_trivial
        

#         x = torch.cat([latent_tokens, critical_visual_tokens, trivial_visual_tokens], dim=1)
#         #print(f"mse loss {F.mse_loss(attn_criticial_weights_sit, attn_critical_weights)}")
#         #attention_map_loss+=F.mse_loss(attn_criticial_weights_sit, attn_critical_weights)*100   # Add this?
#         #attention_map_loss+=F.mse_loss(attn_trivial_weights_sit, attn_trivial_weights)*100   # Add this?
#         ###########################################

#         # for i in range(self.num_vis_tokens*2):
#         #     trivial_visual_token = trivial_visual_tokens[:, i, :]
#         #     trivial_attention_map = attn_trivial_weights[:, i, :]
#         #     x = self.denoising_attention(x, trivial_visual_token, trivial_attention_map)

#         # for i in range(self.num_vis_tokens):
#         #     critical_visual_token = critical_visual_tokens[:, i, :]
#         #     critical_attention_map = attn_critical_weights[:, i, :]
#         #     x = self.denoising_attention(x, critical_visual_token, critical_attention_map)
#         return x, total_attention_map_loss.detach().item()  # option 4
#         #return self.mlp(self.norm2(x)), attention_map_loss   # option 3

# class SiTBlock(nn.Module):
#     """
#     A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, task=None, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#         self.attn = Attention(
#             hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
#             )
#         self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=16, batch_first=True)
#         self.task = task
#         self.num_vis_tokens = NUM_OF_CRITERIA[self.task]
        
#         if "fused_attn" in block_kwargs.keys():
#             self.attn.fused_attn = block_kwargs["fused_attn"]
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.norm_tokens = nn.LayerNorm(768, elementwise_affine=False, eps=1e-6)

#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(
#             in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
#             )
        
#         self.ffn = FFN(hidden_size, hidden_size*4)

#         self.ffn_tokens = FFN_for_SiT(1024, hidden_size*4, hidden_size)

#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )



#     def forward(self, x, visual_tokens, attn_critical_weights, attn_trivial_weights, c):
#         x = self.norm1(x)
#         #print(f"visual_tokens shape {visual_tokens.shape}")
#         critical_visual_tokens = self.norm_tokens(self.ffn_tokens(visual_tokens[:, :self.num_vis_tokens, :]))
#         trivial_visual_tokens= self.norm_tokens(self.ffn_tokens(visual_tokens[:, self.num_vis_tokens:, :]))

#         for i in range(self.num_vis_tokens):
#             critical_token = critical_visual_tokens[:,i,:]
#             critical_weight = attn_critical_weights[:,i,:]
#             x_critical= x * critical_token.unsqueeze(1) * critical_weight.view(x.shape[0], 256, 1)/self.num_vis_tokens
#             x=(x+x_critical)/self.num_vis_tokens

#         for i in range(self.num_vis_tokens):
#             trivial_token = trivial_visual_tokens[:,i,:]
#             trivial_weight = attn_trivial_weights[:,i,:]
#             x_trivial= x * trivial_token.unsqueeze(1) * trivial_weight.view(x.shape[0], 256, 1)/self.num_vis_tokens
#             x=(x+x_trivial)/self.num_vis_tokens
#         return x


class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, task=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=16, batch_first=True)
        self.cross_attn_between_patches = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
        self.cross_attn_vit_output_to_latent = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=16, batch_first=True)
        self.task = task
        self.num_vis_tokens = NUM_OF_CRITERIA[self.task]
        
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_tokens = nn.LayerNorm(768, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        
        self.ffn = FFN(hidden_size, hidden_size*4)

        self.ffn_tokens = FFN_for_SiT(1024, hidden_size*4, hidden_size)
        self.ffn_vit_output = FFN_for_SiT(1024, hidden_size*4, hidden_size)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )



    def forward(self, x, longer_visual_tokens, critical_mask):
        ########################################################## Option to use c
        #print(f"x shape {x.shape}")
        x = self.norm1(x)
        ########################################################
        N, T, D = x.shape
        #vit_l_output = self.norm_tokens(self.ffn_tokens(vit_l_output))
        critical_visual_tokens = self.norm_tokens(self.ffn_tokens(longer_visual_tokens[:, :self.num_vis_tokens, :]))
        #trivial_visual_tokens= self.norm_tokens(self.ffn_tokens(longer_visual_tokens[:, self.num_vis_tokens:, :]))

        
        #critical_denoised_latent, _ = self.cross_attn(x, critical_visual_tokens, critical_visual_tokens)
        #critical_denoised_latent = critical_denoised_latent*critical_mask.view(N, T, 1)
        ######################################################## Option denoise everything 
            ########################################################    Option to use c
        #x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        #critical_visual_tokens = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(self.ffn_tokens(longer_visual_tokens[:, :self.num_vis_tokens, :])), shift_mlp, scale_mlp))
        #trivial_visual_tokens = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(self.ffn_tokens(longer_visual_tokens[:, self.num_vis_tokens:, :])), shift_mlp, scale_mlp))
        
        critical_latent = self.norm1(x*critical_mask.view(N, T, 1))
        trivial_latent = self.norm1(x*(1-critical_mask).view(N, T, 1))
        critical_denoised_latent, _ = self.cross_attn(critical_latent, critical_visual_tokens, critical_visual_tokens)
        x=critical_denoised_latent*critical_mask.view(N, T, 1)+trivial_latent*(1-critical_mask).view(N, T, 1)
        # critical_denoised_vit_l_output, _ = self.cross_attn(vit_l_output, critical_visual_tokens, critical_visual_tokens)
        # critical_denoised_vit_l_output = critical_denoised_vit_l_output*critical_mask.view(N, T, 1)
        # #trivial_denoised_vit_l_output, _ = self.cross_attn(vit_l_output, trivial_visual_tokens, trivial_visual_tokens)
        # trivial_denoised_vit_l_output = trivial_denoised_vit_l_output*(1-critical_mask).view(N, T, 1)
        # vit_l_output=self.norm2(critical_denoised_vit_l_output+trivial_denoised_vit_l_output)
            ########################################################
        ### trivial_denoised_latent, _ = self.cross_attn(x, trivial_visual_tokens, trivial_visual_tokens)
        ### trivial_denoised_latent = trivial_denoised_latent**(1-critical_mask).view(N, T, 1)
        # trivial_latent = trivial_denoised_latent
        ######################################################## Option denoise only critical
        #trivial_latent = x*(1-critical_mask).view(N, T, 1)

        # _, attn_between_patches = self.cross_attn_between_patches(
        #     query=vit_l_output,
        #     key=critical_visual_tokens,
        #     value=critical_visual_tokens,  
        # )  

        # outside_attention = attn_between_patches * (1 - critical_mask.view(N, T, 1))  # Masked attention outside critical areas
        # # Penalize high attention values outside critical mask
        # outside_attention_loss = outside_attention.mean()
        # attention_map_loss=(outside_attention_loss/critical_mask.sum(dim=(-2,-1))).mean()

        # ###x = x+x*(self.norm2(critical_denoised_vit_l_output+trivial_denoised_vit_l_output))

        # #x=x + gate_msa.unsqueeze(1) *(trivial_latent+critical_denoised_latent)
        # # for i in range(self.num_vis_tokens):
        # #     critical_token = critical_visual_tokens[:,i,:]
        # #     critical_weight = attn_critical_weights[:,i,:]
        # #     x_critical= x*critical_mask.view(N, H, W, 1) * self.norm_tokens(self.ffn_tokens(critical_token.unsqueeze(1))) * critical_weight.view(x.shape[0], 256, 1)/self.num_vis_tokens
        # #     x=(x+x_critical)/self.num_vis_tokens
        # x_denoised, _ =self.cross_attn_vit_output_to_latent(x, vit_l_output, vit_l_output)
        # #x=x+self.norm3(x_denoised)
        # x=x+self.norm3(x*vit_l_output)

        return x

class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x):
        x = self.linear(self.norm_final(x)) # option x

        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        #text_embedding_dim=512,
        image_embedding_dim=768,
        task=None,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        self.task = task
        self.num_vis_tokens = NUM_OF_CRITERIA[self.task]

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.x_pure_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        self.image_embedder = nn.Linear(image_embedding_dim, hidden_size)
        num_patches = self.x_embedder.num_patches
        #print(f"num_patches: {num_patches}")
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, task=self.task, **block_kwargs) for _ in range(depth)
        ])
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.y_logits_embedder_in = nn.Linear(num_classes, hidden_size)
        self.y_logits_embedder_out = nn.Linear(hidden_size, num_classes)
        self.ffn = FFN(hidden_size, hidden_size*4)
        self.norm = nn.LayerNorm(hidden_size)
        self.explicd_sigma = torch.nn.Parameter(torch.ones(1))
        self.explicd_attn_sigma = torch.nn.Parameter(torch.ones(1))
        self.denoising_sigma = torch.nn.Parameter(torch.ones(1))
        self.proj_sigma = torch.nn.Parameter(torch.ones(1))
        self.sit_attn_sigma = torch.nn.Parameter(torch.ones(1))
        self.cross_attn_between_patches = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
        self.cross_attn_in_patches = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
        self.patchifyer_model = SiT_Patch_and_Unpatchifier(input_size=input_size, num_classes=num_classes, use_cfg = use_cfg, z_dims = z_dims, encoder_depth=encoder_depth, task = task,**block_kwargs)
        self.patchifyer_model.load_state_dict(torch.load("/home/arsen.abzhanov/Downloads/BioMedia/Thesis/REPA/checkpoints_fr/ISIC/SiT/patchifyer_model.pt")["patchifyer_model"])
        for param in self.patchifyer_model.parameters():
            param.requires_grad = False
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        #nn.init.normal_(self.text_embedder.weight, std=0.02)
        nn.init.normal_(self.image_embedder.weight, std=0.02)
        nn.init.normal_(self.y_logits_embedder_out.weight, std=0.02)
        nn.init.normal_(self.y_logits_embedder_in.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # nn.init.constant_(block.adaLN_modulation_latent[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation_latent[-1].bias, 0)

            # nn.init.constant_(block.adaLN_modulation_tokens[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation_tokens[-1].bias, 0)


        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x, x_pure, x_target,t, y, 
                concept_label, 
                image_embeddings, cls_logits, return_logvar=False,  con_on_explicd_pred=True,
                attn_critical_weights=None, 
                attn_trivial_weights=None,
                vit_l_output=None,
                longer_visual_tokens=None,
                explicid_imgs_noisy_input=None,
                critical_mask = None, 
                trivial_mask = None,
                patchifyer_model=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x=patchifyer_model.patchify_the_latent(x)
        x_pure = patchifyer_model.patchify_the_latent(x_pure)
        # 
        # return samples
        #x_pure = self.x_embedder(x) + self.pos_embed
        #x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        #print(f"x shape {x.shape}")
        #x_pure_completely=self.patchifyer_model(x_pure)
        #x = self.patchifyer_model.patchify_the_latent(x)
        
        #print(f"x shape {x.shape}")
        #x_pure = self.patchifyer_model.patchify_the_latent(x_pure)
        #print(f"self.pos_embed shape {self.pos_embed.shape}")
        #print(f"x shape patchified {x.shape}")
        N, T, D = x.shape
        critical_mask =critical_mask.view(N, T, 1)
        trivial_mask = 1 - critical_mask.view(N, T, 1)

        #mask = critical_mask >= 0.8
        #mask = mask.expand(-1, -1, D)
        #x[mask] = 0
        #let_see_x=patchifyer_model.unpatchify_the_latent(x)
        # print(f"x is {x}")
        # print(f"x max is {torch.max(x)}")
        # print(f"x min is {torch.min(x)}")
        # print(f"=========================================")
        # print(f"critical_mask is {critical_mask}")
        #critical_mask = torch.sigmoid(critical_mask)
        #critical_mask = (critical_mask > 0.5).float()
        #x_pure_unflattened = x_pure.view(N, H, W, D)  # Shape: (B, 16, 16, 768)
        crticial_patches = (torch.zeros_like(x) * critical_mask.view(N, T, 1))  # Shape: (B, 1024, 16, 16)
        trivial_patches = (x * (1-critical_mask).view(N, T, 1))
        # ######################################################## Option denoise only critical
        x = crticial_patches+trivial_patches
        x = x*(torch.ones_like(x)-critical_mask.view(N, T, 1))
        let_see_x=patchifyer_model.unpatchify_the_latent(x)
        #print(f"x shape {x.shape}")

        _, attn_between_patches = self.cross_attn_between_patches(
            query=x_pure,
            key=crticial_patches,
            value=crticial_patches,  
        )

        _, attn_in_patches = self.cross_attn_in_patches(
            query=crticial_patches,
            key=crticial_patches,
            value=crticial_patches,  
        )

        uniform_attention = torch.full_like(attn_in_patches, 1 / T)*critical_mask.view(N, T, 1)  # Ideal uniform distribution
        inside_attn_loss = torch.nn.functional.mse_loss(attn_in_patches*critical_mask.view(N, T, 1), uniform_attention)

        outside_attention = attn_between_patches * (1 - critical_mask.view(N, T, 1))  # Masked attention outside critical areas
        outside_attention_loss = outside_attention.mean()

        attn_map_loss_sit_total=(outside_attention_loss/critical_mask.sum(dim=(-2,-1))).mean() + (inside_attn_loss*critical_mask.sum(dim=(-2,-1))).mean()
        ######################################################## Option denoise everything
        #x=x
        ########################################################
        # timestep and class embedding
        #t_embed = self.t_embedder(t)                   # (N, D)
        #image_embed = self.norm(self.ffn(image_embeddings))
        #y = self.y_embedder(y, self.training)    # (N, D)
        #y_noise = torch.rand_like(y)  # Generate Gaussian noise with the same shape as y
        #minus_embedded_concepts_plus_noise = image_embed
        #c = t_embed + y                                # (N, D)
        #c_concepts=t_embed.unsqueeze(1) + image_embed
        #combined_embeddings_try = torch.cat((x, image_embeddings), dim=1)
        #attn_map_loss_sit_total=torch.tensor(0.0, device=x.device)
        #vit_l_output_for_denoising = vit_l_output.clone().detach()
        for i, block in enumerate(self.blocks):
            #combined_embeddings_try, attn_map_loss= block(combined_embeddings_try, c_concepts, attn_critical_weights, attn_trivial_weights)
            #x, attn_map_loss, vit_l_output, image_embeddings , longer_visual_tokens, explicid_imgs_sit_denoised = block(x, vit_l_output, image_embeddings, longer_visual_tokens, attn_critical_weights, attn_trivial_weights)
            #x, attn_map_loss_current = block(x,  image_embeddings, attn_critical_weights, attn_trivial_weights, c) # option 3
            #combined_embeddings_try, attn_map_loss_current = block(combined_embeddings_try,  image_embeddings, attn_critical_weights, attn_trivial_weights, c) # option 4
            x = block(x,  longer_visual_tokens, critical_mask) # option 4
            ########################################### Option don't change x
            #x, attn_map_loss_current = block(x, longer_visual_tokens, critical_mask, vit_l_output, c)
            ###########################################
            #attn_map_loss_sit_total+=attn_map_loss_current
            #print(f"attn_map_loss {attn_map_loss}")
            if (i + 1) == self.encoder_depth:
                zs = [projector(x[:,:T,:].reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]

        #x = combined_embeddings_try[:,:T,:]
        #x = self.final_layer(x, c_concepts)                # (N, T, patch_size ** 2 * out_channels)
        #x=x_pure
        # x_pure = self.unpatchify(self.final_layer(x)) 
        # x = self.final_layer(x)
        # x = self.unpatchify(x)                   # (N, out_channels, H, W)

        #x = self.patchifyer_model.unpatchify_the_latent(x)
        #x_pure = self.patchifyer_model.unpatchify_the_latent(x_pure)
        x=patchifyer_model.unpatchify_the_latent(x)
        x_pure=patchifyer_model.unpatchify_the_latent(x_pure)
        
        

        sigmas_for_losses = {
            "explicd_sigma": self.explicd_sigma,
            "explicd_attn_sigma": self.explicd_attn_sigma,
            "denoising_sigma":self.denoising_sigma,
            "proj_sigma":self.proj_sigma,
            "sit_attn_sigma": self.sit_attn_sigma
        }

        # y_predicted_logits = self.y_logits_embedder_out(combined_embeddings_try[:,-1:,:]) 
        return let_see_x, x_pure, zs, attn_map_loss_sit_total, sigmas_for_losses

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

class SiT_Patch_and_Unpatchifier(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=768,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        #text_embedding_dim=512,
        image_embedding_dim=768,
        task=None,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        self.task = task
        self.num_vis_tokens = NUM_OF_CRITERIA[self.task]

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.ffn = FFN(hidden_size, hidden_size*4)
        self.norm = nn.LayerNorm(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        #print(f"x shape {x.shape}")
        x = self.final_layer(x)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x
    
    def patchify_the_latent(self, x):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        return x
    
    def unpatchify_the_latent(self, x):
        x = self.final_layer(x)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}

