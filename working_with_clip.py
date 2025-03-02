import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from open_clip import create_model_from_pretrained, get_tokenizer
import torch
import torch.nn as nn
import copy
# # Load the CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device)

# # Define the words
# words = ["asymmetry", "symmetry"]

# # Tokenize the words
# text_inputs = torch.cat([clip.tokenize([word]) for word in words]).to(device)

# # Get text embeddings
# with torch.no_grad():
#     text_features = model.encode_text(text_inputs)

# # Normalize the embeddings
# text_features /= text_features.norm(dim=-1, keepdim=True)

# # Calculate cosine similarity
# similarity = cosine_similarity(text_features.cpu().numpy())
# print(f"Cosine similarity between 'asymmetry' and 'symmetry': {similarity[0, 1]}")

explicid_isic_dict = {
    'color': ['highly variable, often with multiple colors (black, brown, red, white, blue)',   'uniformly tan, brown, or black',  'translucent, pearly white, sometimes with blue, brown, or black areas',   'red, pink, or brown, often with a scale', 'light brown to black',   'pink brown or red', 'red, purple, or blue'],
    'shape': ['irregular', 'round', 'round to irregular', 'variable'],
    'border': ['often blurry and irregular', 'sharp and well-defined', 'rolled edges, often indistinct'],
    'dermoscopic patterns': ['atypical pigment network, irregular streaks, blue-whitish veil, irregular',  'regular pigment network, symmetric dots and globules',  'arborizing vessels, leaf-like areas, blue-gray avoid nests',  'strawberry pattern, glomerular vessels, scale',   'cerebriform pattern, milia-like cysts, comedo-like openings',    'central white patch, peripheral pigment network', 'depends on type (e.g., cherry angiomas have red lacunae; spider angiomas have a central red dot with radiating legs'],
    'texture': ['a raised or ulcerated surface', 'smooth', 'smooth, possibly with telangiectasias', 'rough, scaly', 'warty or greasy surface', 'firm, may dimple when pinched'],
    'symmetry': ['asymmetrical', 'symmetrical', 'can be symmetrical or asymmetrical depending on type'],
    'elevation': ['flat to raised', 'raised with possible central ulceration', 'slightly raised', 'slightly raised maybe thick']

}


model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

prefix_attr_concept_list = ["asymmetrical","symmetrical"]
model.cuda()
tmp_concept_text = tokenizer(prefix_attr_concept_list).cuda()
_, tmp_concept_feats, logit_scale = model(None, tmp_concept_text)
print(tmp_concept_feats.detach().shape)

print(logit_scale * tmp_concept_feats.detach()[0].unsqueeze(0) @ tmp_concept_feats.detach()[1].unsqueeze(1))


tmp_concept_feats /= tmp_concept_feats.norm(dim=-1, keepdim=True)

# Calculate cosine similarity
similarity = cosine_similarity(tmp_concept_feats.detach().cpu().numpy())
print(f"Cosine similarity between 'asymmetry' and 'symmetry': {similarity[0, 1]}") 


# 'asymmetrical', 'symmetrical'
# "regular", "abnormal"
# "uniform", "miscellaneous"
# "Smooth","Rough/Ulcerated"
#  "failed to find Vascular Patterns","There are visible Vascular Patterns"
#  "Flat","Raised"


model_visual_ViT_L = clip.load(f"ViT-L/14")[0].visual
#self.model_visual_custom = VisionTransformer7x7()
#Convert specific layers or parameters to float32
for param in model_visual_ViT_L.parameters():
    param.data = param.data.to(torch.float32)

model_visual_ViT_L.cuda()

#print(clip.load(f"ViT-L/14"))

CONCEPT_LABEL_MAP_ISIC = [
            [3, 0, 0, 3, 3, 0, 2], # AKIEC
            [2, 0, 2, 2, 2, 0, 1], # BCC
            [4, 2, 1, 4, 4, 1, 3], # BKL
            [5, 1, 1, 5, 5, 1, 0], # DF
            [0, 0, 0, 0, 0, 0, 0], # MEL
            [1, 1, 1, 1, 1, 1, 0], # NV
            [6, 3, 1, 6, 1, 2, 0], # VASC
        ]

ISIC={
    "AKIEC":327,
    "BCC":514,
    "BKL":1099,
    "DF":115,
    "MEL":1113,
    "NV":6705,
    "VASC":142
}

print(model_visual_ViT_L)
print(f"Cosine Cosine Cosine  Cosine  Cosine ")
# Access the transformer part of the model
transformer = model_visual_ViT_L.transformer
rgb_tensor = torch.rand(8, 3, 224, 224).cuda()
# Remove the last residual attention block from the Sequential container
new_resblocks = nn.Sequential(*list(transformer.resblocks.children())[:-12])

# Update the new model's transformer with the modified resblocks
transformer.resblocks = new_resblocks
print(model_visual_ViT_L)

# Forward pass through the shared convolutional layer and the first 7 transformer blocks
x = model_visual_ViT_L.conv1(rgb_tensor)  # Convolutional layer
x = x.flatten(2).transpose(1, 2)  # Flatten and transpose
x = model_visual_ViT_L.ln_pre(x)  # Layer normalization
#print(f"x.shape {x.shape}")
#print(f"self.shared_stem.positional_embedding.shape {self.shared_stem.positional_embedding.shape}")
x = x + model_visual_ViT_L.positional_embedding[:x.size(1), :]  # Positional embedding
for i in range(5):
    x = model_visual_ViT_L.transformer.resblocks[i](x)  # First 7 transformer blocks

print(x.shape)
# 1st criteria: 0: MEL,                     1: NV,                      2: BCC,             3: AKIEC,       4: BKL,     5: DF,  6: VASC
# 2nd criteria: 0: MEL, BCC, AKIEC,         1: DF, NV,                  2: BKL,             3: VASC
# 3rd criteria: 0: MEL, AKIEC               1: BKL, DF, NV, VASC        2: BCC
# 4th criteria: 0: MEL                      1: NV                       2: BCC              3: AKIEC        4: BKL,     5: DF,  6: VASC
# 5th criteria: 0: MEL                      1: NV, VASC                 2: BCC              3: AKIEC        4: BKL,     5: DF
# 6th criteria: 0: MEL, AKIEC, BCC          1: BKL, DF, NV              2: VASC
# 7th criteria: 0: MEL, DF, NV, VASC        1: BCC                      2: AKIEC            3: BKL

# 1st criteria: 0: 1113,                        1: 6705,                        2: 514,                 3: 327,       4: 1099,     5: 115,      6: 142
# 2nd criteria: 0: 1113, 514, 327,              1: 115, 6705,                   2: 1099,                3: 142
# 3rd criteria: 0: 1113, 327                    1: 1099, 115, 6705, 142         2: 514
# 4th criteria: 0: 1113                         1: 6705                         2: 514                  3: 327        4: 1099,     5: 115,      6: 142
# 5th criteria: 0: 1113                         1: 6705, 142                    2: 514                  3: 327        4: 1099,     5: 115
# 6th criteria: 0: 1113, 327, 514               1: 1099, 115, 6705              2: 142
# 7th criteria: 0: 1113, 115, 6705, 142        1: 514                           2: 327                  3: 1099


# 1st criteria: 0: 1113,         1: 6705        2: 514                 3: 327       4: 1099     5: 115     6: 142
# 2nd criteria: 0: 1954          1: 6820        2: 1099                3: 142
# 3rd criteria: 0: 1440          1: 8061        2: 514
# 4th criteria: 0: 1113          1: 6705        2: 514                 3: 327       4: 1099     5: 115     6: 142
# 5th criteria: 0: 1113          1: 6847        2: 514                 3: 327       4: 1099     5: 115
# 6th criteria: 0: 1954          1: 7919        2: 142
# 7th criteria: 0: 8075          1: 514         2: 327                 3: 1099

explicid_isic_dict = {
    'color': [ 0.111, 0.669, 0.051, 0.033, 0.11, 0.011, 0.014],
    'shape': [ 0.195, 0.682, 0.11, 0.014],
    'border': [ 0.144, 0.806, 0.051],
    'dermoscopic patterns': [ 0.111, 0.669, 0.051, 0.033, 0.11, 0.011, 0.014],
    'texture': [ 0.111, 0.684, 0.051, 0.033, 0.11, 0.011],
    'symmetry': [ 0.195, 0.791, 0.014],
    'elevation': [ 0.807, 0.051, 0.033, 0.11],

}


dist = [8075, 514, 327, 1099]
s = 10015
d_over_s=[]
for d in dist:
    d_over_s.append(d/s)

for i, w in enumerate(d_over_s):
    print(f"{i}: {round(w/sum(d_over_s), 3)}")