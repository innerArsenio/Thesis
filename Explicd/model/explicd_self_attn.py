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

import torch
import torch.nn as nn

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


class ExpLICD_ViT_L(nn.Module):  
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
                prefix_attr_concept_list = [prefix for concept in attr_concept_list]
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
                prefix_attr_concept_list = [prefix for concept in attr_concept_list]
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