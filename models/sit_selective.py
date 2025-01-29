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

NUM_OF_CRITERIA = {
    'ISIC': 7,
    'ISIC_MINE': 6,
    'IDRID': 5,
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

# class PatchGating(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.linear = nn.Linear(hidden_size, 1)  # predicts gate per patch

#     def forward(self, patches):
#         # patches: shape B,T,D
#         # produce gates in [0,1] range
#         gates = torch.sigmoid(self.linear(patches))  # B,T,1
#         return gates  # soft gates
    
def calculate_sparsity_loss(gates):
    # Encourage sparsity by penalizing the sum of gate values
    sparsity_loss = torch.mean(gates)
    return sparsity_loss


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        #src = src + self.dropout1(src2)
        src = src + src2
        src = self.norm1(src)
        #src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.linear2(self.activation(self.linear1(src)))
        #src = src + self.dropout2(src2)
        src = src + src2
        src = self.norm2(src)
        return src

class PatchGating(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers=2):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_size, 1)  # predicts gate per patch

    def forward(self, patches, T):
        # patches: shape B,T,D
        patches = patches.permute(1, 0, 2)  # T,B,D for transformer
        for layer in self.encoder_layers:
            patches = layer(patches)
        patches = patches.permute(1, 0, 2)  # B,T,D
        gates = torch.sigmoid(self.linear(patches[:,:T,:]))  # B,T,1
        return gates  # soft gates

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
#         self.attn = Attention(
#             hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
#             )
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
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )
#         self.patch_gating = PatchGating(hidden_size)  # Add patch gating module


#     def forward(self, x, c):
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
#             self.adaLN_modulation(c).chunk(6, dim=-1)
#         )

#         T = x.shape[1] - self.num_vis_tokens -1
#         latent_tokens = x[:, :T, :]
#         visual_tokens_crit = x[:, T:T+self.num_vis_tokens, :]
#         y_tokens = x[:, -1:, :]
#         gates = self.patch_gating(latent_tokens)  # B,T,1
#         gated_latent_tokens = latent_tokens * gates  # B,T,D
#         # Perform self-attention on the gated latent tokens and visual tokens
#         x = torch.cat([gated_latent_tokens, visual_tokens_crit, y_tokens], dim=1)
#         x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

#         latent_tokens = x[:, :T, :]
#         visual_tokens_crit = x[:, T:T+self.num_vis_tokens, :]
#         y_tokens = x[:, -1:, :]

#         # Apply MLPs to each modality separately
#         latent_tokens = latent_tokens + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(latent_tokens), shift_mlp, scale_mlp))
#         visual_tokens_crit = visual_tokens_crit + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(visual_tokens_crit), shift_mlp, scale_mlp))
#         y_tokens = y_tokens + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(y_tokens), shift_mlp, scale_mlp))

#         # Concatenate the processed modalities back together
#         combined_embeddings =  torch.cat([gated_latent_tokens, visual_tokens_crit, y_tokens], dim=1)
#         return combined_embeddings
    

class SiTBlock_selection(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, task=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(
        #     hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
        #     )
        self.attn_high = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        # self.attn_low = Attention(
        #     hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
        #     )
        self.task = task
        self.num_vis_tokens = NUM_OF_CRITERIA[self.task]
        
        if "fused_attn" in block_kwargs.keys():
            #self.attn.fused_attn = block_kwargs["fused_attn"]
            self.attn_high.fused_attn = block_kwargs["fused_attn"]
            #self.attn_low.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp_high = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        # self.mlp_low = Mlp(
        #     in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        #     )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        #self.patch_gating = PatchGating(hidden_size)  # Add patch gating module
        

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        
        T = x.shape[1] - self.num_vis_tokens -1
        latent_tokens = x[:, :T, :]
        #print(f"latent_tokens shape {latent_tokens.shape}")
        visual_tokens_crit = x[:, T:T+self.num_vis_tokens, :]
        y_tokens = x[:, -1:, :]

        #low_value_mask = ~self.high_value_mask

        high_value_tokens=latent_tokens
        #high_value_tokens = latent_tokens * high_value_mask  # B,T,D
        #low_value_tokens = latent_tokens * low_value_mask  # B,T,D

        # Perform self-attention on high value tokens concatenated with visual tokens
        high_value_combined = torch.cat([high_value_tokens, visual_tokens_crit, y_tokens], dim=1)
        high_value_combined = high_value_combined + gate_msa.unsqueeze(1) * self.attn_high(modulate(self.norm1(high_value_combined), shift_msa, scale_msa))

        high_value_tokens = high_value_combined[:, :T, :]
        visual_tokens_crit = high_value_combined[:, T:T+self.num_vis_tokens, :]
        y_tokens = high_value_combined[:, -1:, :]

        # Perform self-attention on low value tokens separately
        #low_value_tokens = low_value_tokens + gate_msa.unsqueeze(1) * self.attn_low(modulate(self.norm1(low_value_tokens), shift_msa, scale_msa))

        # Apply MLPs to each modality separately
        high_value_tokens = high_value_tokens + gate_mlp.unsqueeze(1) * self.mlp_high(modulate(self.norm2(high_value_tokens), shift_mlp, scale_mlp))
        visual_tokens_crit = visual_tokens_crit + gate_mlp.unsqueeze(1) * self.mlp_high(modulate(self.norm2(visual_tokens_crit), shift_mlp, scale_mlp))
        y_tokens = y_tokens + gate_mlp.unsqueeze(1) * self.mlp_high(modulate(self.norm2(y_tokens), shift_mlp, scale_mlp))
        #low_value_tokens = low_value_tokens + gate_mlp.unsqueeze(1) * self.mlp_low(modulate(self.norm2(low_value_tokens), shift_mlp, scale_mlp))

        # Combine the processed high and low value tokens back into the original latent structure
        #combined_latent_tokens = high_value_tokens + low_value_tokens
        #print(f"combined_latent_tokens shape {combined_latent_tokens.shape}")
        #high_value_tokens = high_value_tokens * high_value_mask

        # Concatenate the processed modalities back together
        combined_embeddings = torch.cat([high_value_tokens, visual_tokens_crit, y_tokens], dim=1)
        return combined_embeddings


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

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

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
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        self.c1_embedder = LabelEmbedder(7, hidden_size, 0.0)
        self.c2_embedder = LabelEmbedder(4, hidden_size, 0.0)
        self.c3_embedder = LabelEmbedder(3, hidden_size, 0.0)
        self.c4_embedder = LabelEmbedder(7, hidden_size, 0.0)
        self.c5_embedder = LabelEmbedder(6, hidden_size, 0.0)
        self.c6_embedder = LabelEmbedder(3, hidden_size, 0.0)
        self.c7_embedder = LabelEmbedder(4, hidden_size, 0.0)
        #self.text_embedder = nn.Linear(text_embedding_dim, hidden_size)
        self.image_embedder = nn.Linear(image_embedding_dim, hidden_size)
        num_patches = self.x_embedder.num_patches
        #print(f"num_patches: {num_patches}")
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # self.blocks = nn.ModuleList([
        #     SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, task=self.task, **block_kwargs) for _ in range(depth)
        # ])

        self.blocks = nn.ModuleList([
            SiTBlock_selection(hidden_size, num_heads, mlp_ratio=mlp_ratio, task=self.task, **block_kwargs) for _ in range(depth)
        ])
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.y_logits_embedder_in = nn.Linear(num_classes, hidden_size)
        self.y_logits_embedder_out = nn.Linear(hidden_size, num_classes)
        self.patch_gating = PatchGating(hidden_size,4)  # Add patch gating module
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

        nn.init.normal_(self.c1_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.c2_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.c3_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.c4_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.c5_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.c6_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.c7_embedder.embedding_table.weight, std=0.02)

        #nn.init.normal_(self.text_embedder.weight, std=0.02)
        nn.init.normal_(self.image_embedder.weight, std=0.02)
        nn.init.normal_(self.y_logits_embedder_out.weight, std=0.02)
        nn.init.normal_(self.y_logits_embedder_in.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

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
    
    def forward(self, x, x_pure, x_target, t, y, 
                concept_label, 
                image_embeddings, cls_logits, return_logvar=False, con_on_explicd_pred=True):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        x_pure = self.x_embedder(x_pure) + self.pos_embed
        x_target = self.x_embedder(x_target) + self.pos_embed

        N, T, D = x.shape
        #print(f"input shape of x: N {N}, T {T}, D {D}")
        #print("1 3 1")
        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        image_embed = image_embeddings
        # print(y)
        # print(" ")
        # print(torch.max(cls_logits, dim=1)[1])
        y_pred_by_explid=self.y_embedder(torch.max(cls_logits, dim=1)[1], self.training)

        y = self.y_embedder(y, self.training)    # (N, D)
        y_noise = torch.rand_like(y)  # Generate Gaussian noise with the same shape as y

        # con1= self.c1_embedder(concept_label[:,0], self.training)
        # con2= self.c2_embedder(concept_label[:,1], self.training)
        # con3= self.c3_embedder(concept_label[:,2], self.training)
        # con4= self.c4_embedder(concept_label[:,3], self.training)
        # con5= self.c5_embedder(concept_label[:,4], self.training)
        # con6= self.c6_embedder(concept_label[:,5], self.training)
        # con7= self.c7_embedder(concept_label[:,6], self.training)
        # embedded_concepts = torch.stack([con1, con2, con3, con4, con5, con6, con7], dim=1)
        # noise = torch.randn_like(embedded_concepts)  # Generate Gaussian noise with the same shape as embedded_concepts
        # minus_embedded_concepts_plus_noise = embedded_concepts

        concept_noise = torch.randn_like(image_embed)  # Generate Gaussian noise with the same shape as embedded_concepts
        minus_embedded_concepts_plus_noise = image_embed
        #print(f"embedded_concepts shape: {embedded_concepts.shape}")
        #print(y)
        #c = t_embed + y                                # (N, D)
        #c = t_embed + F.normalize(image_embed.mean(dim=1), dim=-1)                                # (N, D)
        #y_input_logits_embedded = self.y_logits_embedder_in(cls_logits)
        #print(f"cls_logits shape: {cls_logits.shape}") 
        #c = t_embed + image_embed[:,0,:] + image_embed[:,1,:] + image_embed[:,2,:] + image_embed[:,3,:] + image_embed[:,4,:] + image_embed[:,5,:] + image_embed[:,6,:]
        # print(f"t_embed shape: {t_embed.shape}")
        # print(f"y_input_logits_embedded shape: {y_input_logits_embedded.shape}")
        if con_on_explicd_pred:
            c = t_embed + y_pred_by_explid
        else:
            c = t_embed + y
        #c = t_embed
        # Concatenate latent image, text embeddings, and conditioning embeddings
        #combined_embeddings=x
        gates = self.patch_gating(torch.cat((x_pure, image_embed), dim=1), T)  # B,T,1
        # Calculate sparsity loss
        #sparsity_loss=0
        sparsity_loss = calculate_sparsity_loss(gates)

        # high_value_mask = gates > 0.5
        # low_value_mask = ~high_value_mask

        # high_value_tokens = x *high_value_mask  # B,T,D

        x= x*gates
        combined_embeddings_try = torch.cat((x, image_embed, y_noise.unsqueeze(1)), dim=1)

        #combined_embeddings_try = torch.cat((high_value_tokens, image_embed, y_noise.unsqueeze(1)), dim=1)

        #combined_embeddings_try = torch.cat((x, image_embed, y_noise.unsqueeze(1)), dim=1)

        #combined_embeddings_try_2 = torch.cat((x, image_embed, concept_noise, y_noise.unsqueeze(1)), dim=1)
        #combined_embeddings_try_3 = torch.cat((x, image_embed, y_noise.unsqueeze(1)), dim=1)
        #combined_embeddings = torch.cat([x, text_embed.unsqueeze(1).expand(-1, T, -1)], dim=1)
        #combined_embeddings = torch.cat([combined_embeddings, image_embed.unsqueeze(1).expand(-1, T, -1)], dim=1)
        #print(c)
        #print("1 3 2")
        for i, block in enumerate(self.blocks):
            #print("koko")
            #x = block(x, c)                      # (N, T, D)
            combined_embeddings_try = block(combined_embeddings_try, c)
            #print("1 3 2 x")
            if (i + 1) == self.encoder_depth:
                #print("1 3 2 y")
                #zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
                #print(f"input shape for projector before reshaping {combined_embeddings.shape}")
                #print(f"input shape for project after reshaping {combined_embeddings.reshape(-1, D).shape}")
                zs = [projector(combined_embeddings_try[:,:T,:].reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
                #zs = [projector(combined_embeddings[:, :T].reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        #print("1 3 2 1")
        x = combined_embeddings_try[:,:T,:]

        # high_value_tokens = x *high_value_mask  # B,T,D
        # low_value_tokens = x_target * low_value_mask  # B,T,D
        # x= low_value_tokens+high_value_tokens
        # x = combined_embeddings[:, :T]
        # text_embed = combined_embeddings[:, T:2*T]
        # image_embed = combined_embeddings[:, 2*T:]
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        #print("1 3 2 2")
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        y_predicted_logits = self.y_logits_embedder_out(combined_embeddings_try[:,-1:,:]) 
        #print("1 3 3")
        return x,image_embed, minus_embedded_concepts_plus_noise, combined_embeddings_try[:,T:T+self.num_vis_tokens,:], y_predicted_logits.squeeze(1), sparsity_loss, zs


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

