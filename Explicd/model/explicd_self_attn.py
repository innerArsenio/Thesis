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

NUM_OF_CRITERIA = {
    'ISIC': 7,
    'ISIC_MINE': 6,

    'IDRID': 5,
    'IDRID_EDEMA': 6,

    'BUSI': 6,
    'BUSI_SOFT': 6
}

NUM_OF_SIMILARITIES = {
    'ISIC': 34,
    'ISIC_MINE': 24,

    'IDRID': 18,
    'IDRID_EDEMA': 15,
    
    'BUSI': 17,
    'BUSI_SOFT': 18
}

def get_prefix(task: str, key: str) -> str:
    if task== "ISIC" or task == "ISIC_MINE":
        return f"this is a dermoscopic image, the {key} of the lesion is "
    elif task == "IDRID" or task=='IDRID_EDEMA':
        return f"this is a fundus image, the {key} of the eye is "
    elif task == "BUSI" or task == "BUSI_SOFT":
        return f"this is an ultrasound image of human breasts, the {key} of the tissue is "
    else:
        raise ValueError(f"Unknown task: {task}. Must be either 'derm' or 'us'")

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