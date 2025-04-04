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
import torchvision.transforms as T

NUM_OF_CRITERIA = {
    'ISIC': 7,
    'ISIC_MINE': 6,
    'ISIC_MINIMAL': 7,
    'ISIC_SOFT': 7,

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



class SiTBlock_smeared(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, task=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        num_heads = 12
        self.cross_attn_crit = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn_trivial = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn_between_patches = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn_vit_output_to_latent = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.task = task
        self.num_vis_tokens = NUM_OF_CRITERIA[self.task]
        
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_tokens = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_vit_output = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        
        self.mlp_crit = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        
        self.mlp_trivial = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.ffn = FFN(hidden_size, hidden_size*4)

        self.ffn_tokens = FFN_for_SiT(1024, hidden_size*4, hidden_size)
        self.proj_tokens = nn.Linear(in_features=1024, out_features=768, bias=False)
        self.ffn_vit_output = FFN_for_SiT(hidden_size, hidden_size*4, hidden_size)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.adaLN_modulation_tokens = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        self.params = [nn.Parameter(torch.randn(1)) for _ in range(self.num_vis_tokens*2)]
        self.lesion_token = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, hidden_size, dtype=torch.float32)))
        self.dregs_token = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, hidden_size, dtype=torch.float32)))



    def forward(self, x, longer_visual_tokens, critical_mask, attn_critical_weights, attn_trivial_weights, vit_l_output, y):
        #print(f"y shape {y.shape}")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(y).chunk(6, dim=-1)
        )

        shift_msa, scale_msa, gate_msa, shift_mlp_crit, shift_mlp_trivial, scale_mlp_crit, scale_mlp_trivial, gate_mlp_crit, gate_mlp_trivial = (
            self.adaLN_modulation_tokens(y).chunk(9, dim=-1)
        )

        #print(f"shift_mlp shape {shift_mlp.shape}")
        #print(f"gate_mlp shape {gate_mlp.shape}")
        ############################################################ Option x is vit_output
        #print(f"x shape {x.shape}")
        T = x.shape[1] - self.num_vis_tokens*2
        B = x.shape[0]
        latent_tokens = x[:, :T, :]
        visual_tokens_crit = x[:, T:T+self.num_vis_tokens, :]
        visual_tokens_trivial = x[:, T+self.num_vis_tokens:, :]
        #print(f"visual_tokens_crit shape {visual_tokens_crit.shape}")
        #print(f"visual_tokens_trivial shape {visual_tokens_trivial.shape}")

        #latent_tokens = latent_tokens + gate_msa.unsqueeze(1) * modulate(self.norm1(latent_tokens), shift_msa, scale_msa)

        #latent_tokens = latent_tokens + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(latent_tokens), shift_mlp, scale_mlp))
        print(f"modulate(self.norm2(visual_tokens_crit), shift_mlp_crit, scale_mlp_crit) shape {modulate(self.norm2(visual_tokens_crit), shift_mlp_crit, scale_mlp_crit).shape}")

        visual_tokens_crit = visual_tokens_crit + gate_mlp_crit.unsqueeze(1) * self.mlp_crit(modulate(self.norm2(visual_tokens_crit), shift_mlp_crit, scale_mlp_crit))
        visual_tokens_trivial = visual_tokens_trivial + gate_mlp_trivial.unsqueeze(1) * self.mlp_trivial(modulate(self.norm3(visual_tokens_trivial), shift_mlp_trivial, scale_mlp_trivial))
        
        lesion_token = self.lesion_token.repeat(B, 1, 1)
        dregs_token = self.dregs_token.repeat(B, 1, 1)

        lesion_token, _ = self.cross_attn_crit(lesion_token, visual_tokens_crit, visual_tokens_crit)
        latent_tokens = latent_tokens + gate_msa.unsqueeze(1) * (lesion_token * attn_critical_weights.mean(dim=1).unsqueeze(1).permute(0, 2, 1))

        #print(f"lesion_token shape {lesion_token.shape}")
        # attn_critical_weights[:, [0, 3, 5], :].mean(dim=1)

        
        dregs_token, _ = self.cross_attn_trivial(dregs_token, visual_tokens_trivial, visual_tokens_trivial)
        latent_tokens = latent_tokens+ gate_msa.unsqueeze(1) * (dregs_token * attn_trivial_weights.mean(dim=1).unsqueeze(1).permute(0, 2, 1))
        #print(f"dregs_token shape {dregs_token.shape}")

        #latent_tokens = latent_tokens + gate_mlp.unsqueeze(1) * self.mlp(self.norm2(latent_tokens))
        #latent_tokens = latent_tokens + self.norm2(latent_tokens  * self.ffn_tokens(vit_l_output))
        #visual_tokens_crit = visual_tokens_crit + gate_mlp.unsqueeze(1) * self.mlp(self.norm2(visual_tokens_crit))
        #x = torch.cat([latent_tokens, visual_tokens_crit], dim=1)
        #return x
        #print(f"latent_tokens shape {latent_tokens.shape}")
        #print(f"visual_tokens_crit shape {visual_tokens_crit.shape}")

        #latent_tokens = latent_tokens + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(latent_tokens), shift_mlp, scale_mlp))
        

        
        #print(f"attn_critical_weights.mean(dim=1).unsqueeze(1) shape {attn_critical_weights.mean(dim=1).unsqueeze(1).shape}")
        #print(f"lesion_token shape {lesion_token.shape}")
        
        
        x = torch.cat([latent_tokens, visual_tokens_crit, visual_tokens_trivial], dim=1)
        return x
        ############################################################
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        T = x.shape[1] - self.num_vis_tokens
        latent_tokens = x[:, :T, :]
        visual_tokens_crit = x[:, T:T+self.num_vis_tokens, :]
        
        # Apply MLPs to each modality separately
        latent_tokens = latent_tokens + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(latent_tokens), shift_mlp, scale_mlp))
        visual_tokens_crit = visual_tokens_crit + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(visual_tokens_crit), shift_mlp, scale_mlp))
        #concept_tokens = concept_tokens + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(concept_tokens), shift_mlp, scale_mlp))

        combined_embeddings = torch.cat([latent_tokens, visual_tokens_crit], dim=1)
        return combined_embeddings
        
        ########################################################## Option to use c
        # #print(f"x shape {x.shape}")
        #x = self.norm_vit_output(x*self.ffn_vit_output(vit_l_output))
        #x = self.norm_vit_output(x*vit_l_output)
        #x=self.norm1(self.norm_vit_output(self.proj_tokens(vit_l_output))*x)
        x = self.norm1(self.mlp(x))
        vit_l_output = self.norm_vit_output(self.ffn_tokens(vit_l_output))
        x = vit_l_output*x
        return x
        ##########################################################
        x = gate_mlp.unsqueeze(1) *self.norm1(self.mlp(x))
        #visual_tokens = self.norm_tokens(self.proj_tokens(longer_visual_tokens))
        ########################################################
        #N, T, D = x.shape
        vit_l_output = self.norm2(self.ffn_tokens(vit_l_output))
        return x+vit_l_output
        #critical_visual_tokens = self.norm_tokens(self.ffn_tokens(longer_visual_tokens[:, :self.num_vis_tokens, :]))
        #trivial_visual_tokens= self.norm_tokens(self.ffn_tokens(longer_visual_tokens[:, self.num_vis_tokens:, :]))

        
        #critical_denoised_latent, _ = self.cross_attn(x, critical_visual_tokens, critical_visual_tokens)
        #critical_denoised_latent = critical_denoised_latent*critical_mask.view(N, T, 1)
        ######################################################## Option denoise everything 
            ########################################################    Option to use c
        #x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        #critical_visual_tokens = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(self.ffn_tokens(longer_visual_tokens[:, :self.num_vis_tokens, :])), shift_mlp, scale_mlp))
        #trivial_visual_tokens = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(self.ffn_tokens(longer_visual_tokens[:, self.num_vis_tokens:, :])), shift_mlp, scale_mlp))
        
        #critical_latent = x*critical_mask.view(N, T, 1)
        #trivial_latent = x*(1-critical_mask).view(N, T, 1)

        #critical_denoised_latent, _ = self.cross_attn(critical_latent, critical_visual_tokens, critical_visual_tokens)
        #x=critical_denoised_latent*critical_mask.view(N, T, 1)+trivial_latent*(1-critical_mask).view(N, T, 1)
        # critical_denoised_vit_l_output, _ = self.cross_attn(vit_l_output, critical_visual_tokens, critical_visual_tokens)
        # critical_denoised_vit_l_output = critical_denoised_vit_l_output*critical_mask.view(N, T, 1)
        # #trivial_denoised_vit_l_output, _ = self.cross_attn(vit_l_output, trivial_visual_tokens, trivial_visual_tokens)
        # trivial_denoised_vit_l_output = trivial_denoised_vit_l_output*(1-critical_mask).view(N, T, 1)
        # vit_l_output=self.norm2(critical_denoised_vit_l_output+trivial_denoised_vit_l_output)
            ########################################################
        #trivial_denoised_latent, _ = self.cross_attn(trivial_latent, trivial_visual_tokens, trivial_visual_tokens)

        #x=critical_denoised_latent*(critical_mask.view(N, T, 1))+trivial_denoised_latent*(1-critical_mask.view(N, T, 1))
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
    
        # Normalize and process visual tokens
        
        ############################################################################
        # Split into critical and trivial tokens
        critical_visual_tokens = visual_tokens[:, :self.num_vis_tokens, :]
        #print(f"critical_visual_tokens shape {critical_visual_tokens.shape}")
        # critical_visual_tokens[:,1,:]=0
        # critical_visual_tokens[:,2,:]=0
        # critical_visual_tokens[:,3,:]=0
        # critical_visual_tokens[:,5,:]=0
        # critical_visual_tokens[:,6,:]=0

        trivial_visual_tokens = visual_tokens[:, self.num_vis_tokens:, :]
        # trivial_visual_tokens[:,1,:]=0
        # trivial_visual_tokens[:,2,:]=0
        # trivial_visual_tokens[:,3,:]=0
        # trivial_visual_tokens[:,5,:]=0
        # trivial_visual_tokens[:,6,:]=0

        # Convert ParameterList to a list of tensors before stacking
        params_list = list(self.params)  # Convert ParameterList to a list
        params_critical = torch.stack(params_list[:7]).view(1, 7, 1, 1)  # First 7 params for critical
        params_trivial = torch.stack(params_list[7:]).view(1, 7, 1, 1)  # Last 7 params for trivial

        # print(f"x.unsqueeze(1) shape {x.unsqueeze(1).shape}")
        # print(f"critical_visual_tokens.unsqueeze(2) shape {critical_visual_tokens.unsqueeze(2).shape}")
        # print(f"attn_critical_weights.unsqueeze(-1) shape {attn_critical_weights.unsqueeze(-1).shape}")
        # print(f"params_critical shape {params_critical.shape}")
        attn_critical_weights = torch.sigmoid(attn_critical_weights / 0.5)
        attn_trivial_weights = torch.sigmoid(attn_trivial_weights / 0.5)
        
        # Compute weighted token interactions in a vectorized way
        x_critical = x.unsqueeze(1) * critical_visual_tokens.unsqueeze(2) * attn_critical_weights.unsqueeze(-1) * params_critical
        x_trivial = x.unsqueeze(1) * trivial_visual_tokens.unsqueeze(2) * attn_trivial_weights.unsqueeze(-1) * params_trivial
        
        # Reduce across the token dimension
        x_critical = x_critical.sum(dim=1)  # Sum over num_vis_tokens
        x_trivial = x_trivial.sum(dim=1)

        return x + gate_msa.unsqueeze(1) * self.norm2(x_critical + x_trivial)  # Combine the contributions

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, task=None, trivial_ratio=0.5, noise_to_crit_map_only=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        num_heads = 12
        self.cross_attn_crit = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn_trivial = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn_between_patches = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn_vit_output_to_latent = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.task = task
        self.num_vis_tokens = NUM_OF_CRITERIA[self.task]
        self.noise_to_crit_map_only = noise_to_crit_map_only
        
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_tokens = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_vit_output = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        
        self.mlp_crit = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        
        self.mlp_trivial = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.ffn = FFN(hidden_size, hidden_size*4)

        self.ffn_tokens = FFN_for_SiT(1024, hidden_size*4, hidden_size)

        self.ffn_crit = FFN_for_SiT(1024, hidden_size*4, 1024)
        self.ffn_trivial = FFN_for_SiT(1024, hidden_size*4, 1024)

        self.norm_crit = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6)
        self.norm_trivial = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6)

        self.proj_tokens = nn.Linear(in_features=1024, out_features=hidden_size, bias=False)
        self.ffn_vit_output = FFN_for_SiT(hidden_size, hidden_size*4, hidden_size)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.adaLN_modulation_crit = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )

        self.adaLN_modulation_trivial = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )

        self.trivial_ratio = trivial_ratio
        self.params = [nn.Parameter(torch.randn(1)) for _ in range(self.num_vis_tokens+int(self.num_vis_tokens*self.trivial_ratio))]
        
        self.lesion_token = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, hidden_size, dtype=torch.float32)))
        self.dregs_token = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, hidden_size, dtype=torch.float32)))

        self.latent_shift = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.latent_scale = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

        self.crit_shift = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.crit_scale = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

        self.triv_shift = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.triv_scale = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))



    def forward(self, x, longer_visual_tokens, critical_mask, attn_critical_weights, attn_trivial_weights, vit_l_output, y, use_actual_latent_of_the_images):

        B = x.shape[0]
        latent_tokens = x
        visual_tokens_crit = longer_visual_tokens[:, :self.num_vis_tokens, :]
        visual_tokens_trivial = longer_visual_tokens[:, self.num_vis_tokens:, :]

        shift_latent, scale_latent, shift_crit, scale_crit, shift_triv, scale_triv = (
            self.adaLN_modulation(y).chunk(6, dim=-1)
        )


        visual_tokens_crit = self.proj_tokens(self.norm_crit(self.ffn_crit(visual_tokens_crit)))

        if not self.noise_to_crit_map_only:
            visual_tokens_trivial = self.proj_tokens(self.norm_trivial(self.ffn_trivial(visual_tokens_trivial)))

        visual_tokens_crit =  modulate(self.norm2(visual_tokens_crit), shift_crit, scale_crit)

        if not self.noise_to_crit_map_only:
            visual_tokens_trivial =  modulate(self.norm3(visual_tokens_trivial), shift_triv, scale_triv)

        attn_critical_weights = attn_critical_weights/torch.max(attn_critical_weights, dim=-1, keepdim=True).values

        if not self.noise_to_crit_map_only:
            attn_trivial_weights = attn_trivial_weights/torch.max(attn_trivial_weights, dim=-1, keepdim=True).values

        params_critical = torch.stack(self.params[:self.num_vis_tokens]).view(1, self.num_vis_tokens, 1).to(device=x.device)  # Shape (1, 7, 1)

        if not self.noise_to_crit_map_only:
            params_trivial = torch.stack(self.params[self.num_vis_tokens:]).view(1, int(self.num_vis_tokens*self.trivial_ratio), 1).to(device=x.device)   # Shape (1, 7, 1)
    
        critical_contributions_all = (visual_tokens_crit.unsqueeze(2) * attn_critical_weights.unsqueeze(-1) * params_critical.unsqueeze(-1))
        #critical_contributions_all[:, 1:, :] = 0
        critical_contributions = critical_contributions_all.mean(dim=1)

        if not self.noise_to_crit_map_only:
            trivial_contributions = (visual_tokens_trivial.unsqueeze(2) * attn_trivial_weights.unsqueeze(-1) * params_trivial.unsqueeze(-1)).mean(dim=1)

        # shift_latent = self.latent_shift.repeat(B, 1)
        # scale_latent = self.latent_scale.repeat(B, 1)

        # shift_crit = self.crit_shift.repeat(B, 1)
        # scale_crit = self.crit_scale.repeat(B, 1)

        # shift_triv = self.triv_shift.repeat(B, 1)
        # scale_triv = self.triv_scale.repeat(B, 1)

        #latent_tokens = shift_latent.unsqueeze(1)*latent_tokens + latent_shift_msa.unsqueeze(1) + critical_contributions + trivial_contributions
        if not self.noise_to_crit_map_only:
            latent_tokens =  modulate(self.norm1(latent_tokens), shift_latent, scale_latent) +  critical_contributions + trivial_contributions
        else:
            latent_tokens = modulate(self.norm1(latent_tokens), shift_latent, scale_latent) +  critical_contributions
        #latent_tokens = modulate(self.norm1(latent_tokens), shift_latent, scale_latent) + modulate(self.norm2(critical_contributions), shift_crit, scale_crit) + modulate(self.norm3(trivial_contributions), shift_triv, scale_triv) 
        x=latent_tokens
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
        class_dropout_prob=0.0,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        #text_embedding_dim=512,
        image_embedding_dim=768,
        task=None,
        denoise_patches=0,
        use_actual_latent_of_the_images=1,
        trivial_ratio=0.5,
        noise_to_crit_map_only=False,
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
        self.denoise_patches = denoise_patches
        self.use_actual_latent_of_the_images = use_actual_latent_of_the_images
        self.trivial_ratio = trivial_ratio
        self.noise_to_crit_map_only = noise_to_crit_map_only

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
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, task=self.task, trivial_ratio = self.trivial_ratio, noise_to_crit_map_only=self.noise_to_crit_map_only, **block_kwargs) for _ in range(depth)
        ])
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.y_logits_embedder_in = nn.Linear(num_classes, hidden_size)
        self.y_logits_embedder_out = nn.Linear(hidden_size, num_classes)
        self.ffn = FFN(hidden_size, hidden_size*4)
        self.ffn_vit_output = FFN_for_SiT(1024, hidden_size*4, hidden_size)
        self.norm_vit_output = nn.LayerNorm(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.explicd_sigma = torch.nn.Parameter(torch.ones(1))
        self.explicd_attn_sigma = torch.nn.Parameter(torch.ones(1))
        self.denoising_sigma = torch.nn.Parameter(torch.ones(1))
        self.proj_sigma = torch.nn.Parameter(torch.ones(1))
        self.y_learned_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, hidden_size, dtype=torch.float32)))
        self.y_learned_tokens.requires_grad = True
        self.cross_attn_y_tokens = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=12, batch_first=True)
        self.sit_attn_sigma = torch.nn.Parameter(torch.ones(1))
        self.cross_attn_between_patches = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
        self.cross_attn_in_patches = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
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
                patchifyer_model=None,
                highlight_the_critical_mask=False,
                patches = None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x_pure = patchifyer_model.patchify_the_latent(x_pure)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t_embed = self.t_embedder(t)                   # (N, D)
        # y = self.y_embedder(y, self.training)    # (N, D)

        # if self.use_actual_latent_of_the_images==0:
        #     y = y + t_embed
        attn_map_loss_sit_total=torch.tensor(0.0, device=x.device)
        N, T, D = x.shape
        #H = W = int(T ** 0.5)
        y = t_embed
        if critical_mask is not None:
            critical_mask =critical_mask.view(N, T, 1)

        # y_learned_tokens = self.y_learned_tokens.repeat(N, 1, 1)
        # y_learned_tokens, _ = self.cross_attn_y_tokens(y_learned_tokens, image_embeddings, image_embeddings)
        #image_embeddings = image_embeddings[:, :self.num_vis_tokens, :]
        #x = torch.cat((x, image_embeddings), dim=1)
        if self.use_actual_latent_of_the_images==0:
            use_actual_latent_of_the_images = False
        elif self.use_actual_latent_of_the_images==1:
            use_actual_latent_of_the_images = True

        for i, block in enumerate(self.blocks):
            x = block(x, longer_visual_tokens, critical_mask, attn_critical_weights, attn_trivial_weights, vit_l_output, y, use_actual_latent_of_the_images) # option 4
            if (i + 1) == self.encoder_depth:
                zs = [projector(x[:,:T,:].reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]

        x = x[:,:T,:]
        if self.denoise_patches==1:
            patches = x
        x = self.final_layer(x)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        if highlight_the_critical_mask:
            # fix thi
            x_critical_removed = x_pure*(torch.ones_like(x_pure)-critical_mask.view(N, T, 1)) + 0.001*torch.rand_like(x_pure)*critical_mask.view(N, T, 1)
            x_critical_removed = patchifyer_model.unpatchify_the_latent(x_critical_removed)

        x_pure=patchifyer_model.unpatchify_the_latent(x_pure)
        
        

        sigmas_for_losses = {
            "explicd_sigma": self.explicd_sigma,
            "explicd_attn_sigma": self.explicd_attn_sigma,
            "denoising_sigma":self.denoising_sigma,
            "proj_sigma":self.proj_sigma,
            "sit_attn_sigma": self.sit_attn_sigma
        }

        if highlight_the_critical_mask:
            return patches, x, x_pure, x_critical_removed
        else:
            return patches, x, x_pure, zs, attn_map_loss_sit_total, sigmas_for_losses

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
    return SiT(depth=9, hidden_size=768, decoder_hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def SiT_B_2_patches(**kwargs):
    return SiT(depth=12, hidden_size=588, decoder_hidden_size=588, patch_size=2, num_heads=12, **kwargs)

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
    'SiT-B/2':  SiT_B_2, 'SiT-B/2_patches':  SiT_B_2_patches,  'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}

