import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from sklearn.metrics import balanced_accuracy_score
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

CONCEPT_LABEL_MAP_BUSI = [
    # Benign
    [1, 1, 1, 1, 1, 0],
    # Malignant
    [2, 2, 2, 2, 2, 1],  
    # Normal
    [0, 0, 0, 0, 0, 0],  
]

# CONCEPT_LABEL_MAP_BUSI = [
#     # Normal
#     [0, 0, 0, 0, 0, 0], 
#     # Benign
#     [1, 1, 1, 1, 1, 0],
#     # Malignant
#     [2, 2, 2, 2, 2, 1]
# ]

CONCEPT_LABEL_MAP_ISIC_MINIMAL = [
            [0, 0, 0, 0, 0, 0, 0], # AKIEC
            [1, 1, 1, 1, 1, 1, 1], # BCC
            [2, 2, 2, 2, 2, 2, 2], # BKL
            [3, 3, 3, 3, 3, 3, 3], # DF
            [4, 4, 4, 4, 4, 4, 4], # MEL
            [5, 5, 5, 5, 5, 5, 5], # NV
            [6, 6, 6, 6, 6, 6, 6], # VASC
        ]

# Actinic Keratoses, Basal Cell Carcinoma, Benign Keratosis-like Lesions, Dermatofibroma, Melanoma, Melanocytic Nevus, Vascular Lesions
CONCEPT_LABEL_MAP_ISIC_SOFT_SMOOTH = [
    # Actinic Keratoses
    [[0.0, 0.0, 0.0, 0.85, 0.0, 0.15, 0.0],  # color: red, pink, or brown, often with a scale
     [0.85, 0.1, 0.05, 0.0],  # shape: irregular
     [0.75, 0.15, 0.1],  # border: often blurry and irregular
     [0.0, 0.0, 0.0, 0.75, 0.15, 0.1, 0.0],  # dermoscopic patterns: strawberry pattern, glomerular vessels, scale
     [0.0, 0.0, 0.0, 0.85, 0.15, 0.0],  # texture: rough, scaly
     [0.85, 0.1, 0.05],  # symmetry: asymmetrical
     [0.05, 0.1, 0.85, 0.0]],  # elevation: slightly raised, maybe thick

    # Basal Cell Carcinoma
    [[0.0, 0.0, 0.75, 0.15, 0.0, 0.0, 0.1],  # color: translucent, pearly white, sometimes with blue, brown, or black areas
     [0.0, 0.85, 0.15, 0.0],  # shape: round
     [0.05, 0.1, 0.85],  # border: rolled edges, often indistinct
     [0.0, 0.0, 0.8, 0.15, 0.05, 0.0, 0.0],  # dermoscopic patterns: arborizing vessels, leaf-like areas, blue-gray nests
     [0.0, 0.1, 0.85, 0.05, 0.0, 0.0],  # texture: smooth, possibly with telangiectasias
     [0.05, 0.85, 0.1],  # symmetry: symmetrical
     [0.05, 0.85, 0.1, 0.0]],  # elevation: raised with possible central ulceration

    # Benign Keratosis-like Lesions
    [[0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0],  # color: light brown to black
     [0.0, 0.3, 0.7, 0.0],  # shape: round to irregular
     [0.05, 0.9, 0.05],  # border: sharp and well-defined
     [0.0, 0.0, 0.0, 0.0, 0.85, 0.15, 0.0],  # dermoscopic patterns: cerebriform pattern, milia-like cysts, comedo-like openings
     [0.0, 0.0, 0.0, 0.85, 0.15, 0.0],  # texture: warty or greasy surface
     [0.05, 0.9, 0.05],  # symmetry: symmetrical
     [0.05, 0.1, 0.85, 0.0]],  # elevation: slightly raised

    # Dermatofibroma
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1],  # color: pink, brown, or red
     [0.05, 0.1, 0.85, 0.0],  # shape: variable
     [0.05, 0.85, 0.1],  # border: sharp and well-defined
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.15],  # dermoscopic patterns: central white patch, peripheral pigment network
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.85],  # texture: firm, may dimple when pinched
     [0.05, 0.05, 0.9],  # symmetry: can be symmetrical or asymmetrical
     [0.05, 0.05, 0.9, 0.0]],  # elevation: slightly raised

    # Melanoma
    [[0.9, 0.05, 0.0, 0.05, 0.0, 0.0, 0.0],  # color: highly variable, often with multiple colors
     [0.9, 0.05, 0.05, 0.0],  # shape: irregular
     [0.9, 0.05, 0.05],  # border: often blurry and irregular
     [0.9, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],  # dermoscopic patterns: atypical pigment network, irregular streaks, blue-whitish veil
     [0.9, 0.05, 0.05, 0.0, 0.0, 0.0],  # texture: a raised or ulcerated surface
     [0.9, 0.05, 0.05],  # symmetry: asymmetrical
     [0.9, 0.05, 0.05, 0.0]],  # elevation: flat to raised

    # Melanocytic Nevus
    [[0.0, 0.85, 0.0, 0.0, 0.1, 0.05, 0.0],  # color: uniformly tan, brown, or black
     [0.05, 0.9, 0.05, 0.0],  # shape: round
     [0.05, 0.9, 0.05],  # border: sharp and well-defined
     [0.0, 0.85, 0.1, 0.05, 0.0, 0.0, 0.0],  # dermoscopic patterns: regular pigment network, symmetric dots and globules
     [0.0, 0.9, 0.1, 0.0, 0.0, 0.0],  # texture: smooth
     [0.1, 0.85, 0.05],  # symmetry: symmetrical
     [0.05, 0.1, 0.85, 0.0]],  # elevation: slightly raised

    # Vascular Lesions
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # color: red, purple, or blue
     [0.05, 0.1, 0.85, 0.0],  # shape: variable
     [0.05, 0.9, 0.05],  # border: sharp and well-defined
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # dermoscopic patterns: depends on type (e.g., cherry angiomas have red lacunae)
     [0.0, 0.9, 0.1, 0.0, 0.0, 0.0],  # texture: smooth
     [0.05, 0.9, 0.05],  # symmetry: symmetrical
     [0.05, 0.1, 0.85, 0.0]],  # elevation: slightly raised
]

CONCEPT_LABEL_MAP_BUSI_SOFT_SMOOTH = [
    # Benign
    [[0.2, 0.6, 0.2], [0.2, 0.6, 0.1, 0.1], [0.0, 0.8, 0.1, 0.1], [0.6, 0.1, 0.3], [0.1, 0.9] , [0.8, 0.2]],
    # Malignant
    [[0.0, 0.1, 0.9], [0.0, 0.1, 0.2, 0.7], [0.0, 1.0, 0.0, 0.0], [0.1, 0.8, 0.1], [0.8, 0.2], [0.2, 0.8]],  
    # Normal
    [[0.8, 0.2, 0.0], [1.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.7, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],  
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

CONCEPT_LABEL_MAP_DICT = {
    'ISIC': CONCEPT_LABEL_MAP_ISIC,
    'ISIC_MINE': CONCEPT_LABEL_MAP_ISIC_MINE,
    'ISIC_MINIMAL': CONCEPT_LABEL_MAP_ISIC_MINIMAL,
    'ISIC_SOFT':CONCEPT_LABEL_MAP_ISIC_SOFT_SMOOTH,
    'IDRID': CONCEPT_LABEL_MAP_IDRID,
    'BUSI': CONCEPT_LABEL_MAP_BUSI,
    'BUSI_SOFT': CONCEPT_LABEL_MAP_BUSI_SOFT_SMOOTH,
    'IDRID_EDEMA': CONCEPT_LABEL_MAP_IDRID_EDEMA
}



CE_WEIGHTS = {
    'ISIC': [ 0.134, 0.084, 0.039, 0.389, 0.039, 0.0065, 0.305],  # train weights
    #'ISIC': [ 0.24, 0.11, 0.05, 0.23, 0.06, 0.01, 0.29],  # test weights
    'ISIC_MINE': [ 0.134, 0.084, 0.039, 0.389, 0.039, 0.0065, 0.305],  # train weights
    'ISIC_MINIMAL': [ 0.134, 0.084, 0.039, 0.389, 0.039, 0.0065, 0.305],  # train weights
    'ISIC_SOFT': [ 0.134, 0.084, 0.039, 0.389, 0.039, 0.0065, 0.305],  # train weights

    'IDRID': [ 0.076, 0.506, 0.074, 0.137, 0.207],
    'IDRID_EDEMA': [0.1606, 0.6935, 0.1458],

    'BUSI': [ 0.157, 0.327, 0.516],
    'BUSI_SOFT': [ 0.157, 0.327, 0.516]
}


explicid_isic_dict_weights = {
    'color': [ 0.04, 0.007, 0.086, 0.135, 0.04, 0.383, 0.31],
    'shape': [ 0.059, 0.017, 0.106, 0.818],
    'border': [ 0.251, 0.045, 0.704],
    'dermoscopic patterns': [ 0.04, 0.007, 0.086, 0.135, 0.04, 0.383, 0.31],
    'texture': [ 0.057, 0.009, 0.124, 0.195, 0.058, 0.556],
    'symmetry': [ 0.067, 0.016, 0.917],
    'elevation': [ 0.021, 0.322,0.507, 0.151],

}

explicid_idrid_dict_weights = {
    'microaneurysms': [0.102, 0.685, 0.101, 0.111],
    'hemorrhages': [0.102, 0.685, 0.101, 0.111],
    'exudates': [0.102, 0.685, 0.101, 0.111],
    'neovascularization': [0.119, 0.881],
    'macular edema': [0.136, 0.154, 0.283, 0.427],
}

explicid_idrid_edema_dict_weights = {
    'hard exudates': [0.161, 0.694, 0.146],
    'retinal thickening': [0.571, 0.429],
    'microaneurysms': [0.571,  0.429],
    'hemorrhages': [0.161, 0.694, 0.146],
    'cotton wool spots': [0.472, 0.528],
    'vascular abnormalities': [0.161, 0.694, 0.146]
}

explicid_busi_dict_weights = {
    'shape': [0.516, 0.157, 0.327],
    'margin': [0.516, 0.157, 0.327],
    'echo': [0.516, 0.157, 0.327],
    'posterior': [0.516, 0.157, 0.327],
    'calcifications': [0.516, 0.157, 0.327],
    'orientation': [0.269, 0.731]
}


CONCEPT_CE_WEIGHTS = {
    'ISIC': explicid_isic_dict_weights,
    'IDRID': explicid_idrid_dict_weights,
    'IDRID_EDEMA': explicid_idrid_edema_dict_weights,
    'BUSI': explicid_busi_dict_weights
}


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

# Compute the contrastive loss
def contrastive_loss(embedding1, embedding2, positive=True):
    similarity = cosine_similarity(embedding1, embedding2, dim=-1)
    if positive:
        return 1 - similarity.mean()
    else:
        return similarity.mean()

def label_smoothing(one_hot_labels, epsilon=0.1):
    num_classes = one_hot_labels.size(1)
    smooth_labels = one_hot_labels * (1 - epsilon) + (1 - one_hot_labels) * (epsilon / (num_classes - 1))
    return smooth_labels

def calculate_logits_similarity_loss(cls_logits, cls_logits_dict, labels, num_classes=7):
    similarity_loss = 0.0
    for class_idx in range(num_classes):
        # Get indices where labels == class_idx
        class_mask = (labels == class_idx)
        if not class_mask.any():
            continue
            
        # Get logits for samples of this class
        original_logits = cls_logits[class_mask]
        class_specific_logits = cls_logits_dict[class_idx][class_mask]
        
        # Calculate MSE loss between the logits
        class_similarity_loss = F.mse_loss(original_logits, class_specific_logits)
        similarity_loss += class_similarity_loss
    
    return similarity_loss

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            task = None,
            do_logits_similarity=True,
            concept_hardness="soft_equal",
            cls_loss_epoch=0,
            noise_to_crit_map_only = False
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.task = task
        #self.lesion_weight = torch.FloatTensor([ 0.134, 0.084, 0.039, 0.389, 0.039, 0.0065, 0.305]).cuda()   ISIC
        #self.lesion_weight = torch.FloatTensor([ 0.157, 0.327, 0.516]).cuda()  BUSI
        self.ce_weight = torch.FloatTensor(CE_WEIGHTS[self.task]).cuda()
        self.cls_criterion = nn.CrossEntropyLoss(weight=self.ce_weight).cuda()
        self.concept_label_map = CONCEPT_LABEL_MAP_DICT[task]
        self.do_logits_similarity=do_logits_similarity
        self.concept_hardness=concept_hardness
        self.cls_loss_epoch=cls_loss_epoch
        self.noise_to_crit_map_only = noise_to_crit_map_only
        

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, latent_raw_image, model_kwargs=None, zs=None, labels=None, explicid=None, explicid_imgs_list= None, epoch=None,  explicd_only=0, do_sit=False, do_pretraining_the_patchifyer=False, patchifyer_model=None, denoise_patches=0, use_actual_latent_of_the_images=0, explicid_imgs_wo_norm_list=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        images = latent_raw_image
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)    
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        use_te_loss=False
        #print(f"images shape: {images.shape}")
        # if use_actual_latent_of_the_images==0:
        #     model_input = alpha_t * images + sigma_t * noises
        # elif use_actual_latent_of_the_images==1:
        #     model_input = torch.randn_like(images)

        # if self.prediction == 'v' and use_actual_latent_of_the_images==0:
        #     model_target = d_alpha_t * images + d_sigma_t * noises
        # elif self.prediction == 'v' and use_actual_latent_of_the_images==1:
        #     model_target = images
        # else:
        #     raise NotImplementedError() # TODO: add x or eps prediction

        denoising_loss, proj_loss, explicid_loss, cosine_loss, sit_cls_loss, logits_similarity_loss, attn_map_loss_sit_total = [torch.tensor(0.0, device=images.device) for _ in range(7)] 
        contr_loss= torch.tensor(0.0, device=images.device)
        attn_explicd_loss=torch.tensor(0.0, device=images.device)
        loss_cls_criteria_only = torch.tensor(0.0, device=images.device)
        loss_cls_refined = torch.tensor(0.0, device=images.device)
        sit_cls_loss = torch.tensor(0.0, device=images.device)
        loss_cls_with_te = torch.tensor(0.0, device=images.device)
        te_loss = torch.tensor(0.0, device=images.device)
        sigmas_for_losses = {}
        for imgs_indx, (explicid_imgs,explicid_imgs_wo_norm)  in enumerate(zip(explicid_imgs_list,explicid_imgs_wo_norm_list)):
            # if do_pretraining_the_patchifyer:
            #     model_output = model(model_input)
            #     processing_loss = mean_flat((model_output - model_target) ** 2)
            #     return processing_loss, [torch.tensor(0.0, device=images.device)]*15
            # with torch.no_grad():
            #     explicid_imgs_latents=patchifyer_model.patchify_the_latent(latent_raw_image)
            explicd_return_dict = explicid(explicid_imgs, explicid_imgs_latents=None, imgs_wo_norm=None)
            patches = explicd_return_dict["patches"]
            cls_logits = explicd_return_dict["cls_logits"]
            cls_with_te_logits = explicd_return_dict["cls_with_te_logits"]
            te_loss = 0*explicd_return_dict["te_loss"]
            image_logits_dict = explicd_return_dict["image_logits_dict"]
            agg_critical_tokens = explicd_return_dict["agg_critical_visual_tokens_for_SiT"]
            agg_trivial_tokens = explicd_return_dict["agg_trivial_visual_tokens_for_SiT"]
            attn_explicd_loss_dict = explicd_return_dict["attn_explicd_loss_dict"]
            overlap_loss = explicd_return_dict["overlap_loss"]
            attn_critical_weights = explicd_return_dict["attn_critical_weights"]
            attn_trivial_weights = explicd_return_dict["attn_trivial_weights"]
            vit_l_output = explicd_return_dict["vit_l_output"]
            agg_critical_visual_tokens = explicd_return_dict["agg_critical_visual_tokens"]
            agg_trivial_visual_tokens = explicd_return_dict["agg_trivial_visual_tokens"]
            cnn_logits_critical = explicd_return_dict["cnn_logits_critical"]
            cnn_logits_trivial = explicd_return_dict["cnn_logits_trivial"]
            critical_mask = explicd_return_dict["critical_mask"]
            trivial_mask = explicd_return_dict["trivial_mask"]

            #attn_critical_weights_int = attn_critical_weights[:,0,:].unsqueeze(1)
            attn_critical_weights_int = attn_critical_weights
            normalized_attention = attn_critical_weights_int / attn_critical_weights_int.max(dim=-1, keepdim=True).values
            summed_attention = normalized_attention.sum(dim=1)
            clipped_attention = torch.clamp(summed_attention, min=0.0, max=1.0)
            B, T, D = vit_l_output.shape  # B: batch size, T: num_patches (256), D: patch_dim (1024)
            H = W = int(T ** 0.5)  # Assuming T is a perfect square, H = W = 16
            clipped_attention = clipped_attention.view(B, H, W)
            clipped_attention = clipped_attention.unsqueeze(1)

            if torch.isnan(te_loss):
                te_loss = torch.tensor(0.0).to(device=images.device)

            if self.noise_to_crit_map_only:
                #print(f"clipped_attention shape: {clipped_attention.shape}")
                #print(f"images shape: {images.shape}")
                images = images*clipped_attention
                #print(f"images shape after multiplication: {images.shape}")


            if use_actual_latent_of_the_images==0:
                model_input = alpha_t * images + sigma_t * noises
            elif use_actual_latent_of_the_images==1:
                model_input = torch.randn_like(images)

            if self.prediction == 'v' and use_actual_latent_of_the_images==0:
                model_target = d_alpha_t * images + d_sigma_t * noises
            elif self.prediction == 'v' and use_actual_latent_of_the_images==1:
                model_target = images
            else:
                raise NotImplementedError() # TODO: add x or eps prediction

            if self.task =="IDRID":
                attn_explicd_loss+=attn_explicd_loss_dict["black_crit_loss"]
                #print(f"labels shape: {labels.shape}")
                # Check if each element is in {2, 3, 4}
                mask = torch.isin(labels, torch.tensor([2, 3, 4],device=images.device))
                # # Set the values to 1 if in {2, 3, 4}, else set to 0 
                tensor_viable = mask.int().to(device=images.device).unsqueeze(1).unsqueeze(2)
                # #print(f"tensor_viable shape: {tensor_viable.shape}") 
                # #print(f"attn_explicd_loss_dict['color_loss'] shape: {attn_explicd_loss_dict["color_loss"].shape}")
                #attn_explicd_loss+=torch.mean(tensor_viable*attn_explicd_loss_dict["color_loss"])

                #attn_explicd_loss+=0.5*attn_explicd_loss_dict["smooth_loss"]*10e9
                attn_explicd_loss+=attn_explicd_loss_dict["overlap_loss"]*10e3
            elif self.task =="ISIC":
                attn_explicd_loss+=attn_explicd_loss_dict["var_loss"]*10e7
                attn_explicd_loss+=0.5*attn_explicd_loss_dict["smooth_loss"]*10e9
                attn_explicd_loss+=attn_explicd_loss_dict["overlap_loss"]*10e3

                # mask = torch.isin(labels, torch.tensor([5],device=images.device))
                # tensor_viable = mask.int().to(device=images.device).unsqueeze(1).unsqueeze(2)
                # attn_explicd_loss+=0*torch.mean(tensor_viable*attn_explicd_loss_dict["color_loss"])

                attn_explicd_loss+=torch.mean(attn_explicd_loss_dict["color_loss"])
                
                #lesion_labels = F.one_hot(labels, num_classes=len(self.concept_label_map)).unsqueeze(-1)
                #M = attn_explicd_loss_dict['masks'] * lesion_labels  # Element-wise selection (B, 7, 256)
                #M_combined = M.sum(dim=1)  # Reduce to (B, 256) by summing over active classes
                #attn_explicd_loss+= F.mse_loss(attn_explicd_loss_dict["crit_dot"][:, 0], M_combined)  # Ensure 1st token matches lesion-relevant patches
                #attn_explicd_loss+= F.mse_loss(normalized_attention[:, 0, :], M_combined)  # Ensure 1st token matches lesion-relevant patches
                #attn_explicd_loss+=10*attn_explicd_loss_dict["var_loss_other"]
                #attn_explicd_loss*=10
                #attn_explicd_loss*=0

                attention_maps_norm = F.normalize(attn_critical_weights, p=2, dim=-1)
                
                # Compute the pairwise cosine similarity between all attention maps
                cosine_sim = torch.bmm(attention_maps_norm, attention_maps_norm.transpose(1, 2))
                
                # Create a mask to ignore the diagonal (self-similarity), since we don't want the map to compare to itself
                mask = torch.eye(attn_critical_weights.size(1), device=images.device).unsqueeze(0)  # Shape (B, N_c, N_c)
                
                # Apply the mask (we'll set diagonal values to -1 so they are ignored in the loss calculation)
                cosine_sim = cosine_sim.masked_fill(mask == 1, -1)

                # Now, calculate the loss as the negative average cosine similarity between all pairs
                #attn_explicd_loss = 10*(1 - cosine_sim.mean())
            if explicd_only==0:
                ##################  Option use trivial tokens too
                agg_visual_tokens = torch.cat((agg_critical_tokens, agg_trivial_tokens), dim=1)
                ##################  Option use critical tokens only
                #agg_visual_tokens = agg_critical_tokens
                ##################
                longer_visual_tokens = torch.cat((agg_critical_visual_tokens, agg_trivial_visual_tokens), dim=1)
                vit_l_output_input = vit_l_output


            if self.concept_hardness!="soft_smarter":
                concept_label = torch.tensor([self.concept_label_map[label] for label in labels])
                concept_label = concept_label.long().cuda()
            else:
                if self.task == "BUSI_SOFT":
                    concept_label = [self.concept_label_map[label] for label in labels]
                    concept_label_smooth_dict = {
                        'shape': torch.tensor([inner_list[0] for inner_list in concept_label]).cuda(),
                        'margin': torch.tensor([inner_list[1] for inner_list in concept_label]).cuda(),
                        'echo pattern': torch.tensor([inner_list[2] for inner_list in concept_label]).cuda(),
                        'posterior features': torch.tensor([inner_list[3] for inner_list in concept_label]).cuda(),
                        'calcifications': torch.tensor([inner_list[4] for inner_list in concept_label]).cuda(),
                        'orientation': torch.tensor([inner_list[5] for inner_list in concept_label]).cuda()

                    }
                elif self.task =="ISIC_SOFT":
                    concept_label = [self.concept_label_map[label] for label in labels]
                    concept_label_smooth_dict = {
                        'color': torch.tensor([inner_list[0] for inner_list in concept_label]).cuda(),
                        'shape': torch.tensor([inner_list[1] for inner_list in concept_label]).cuda(),
                        'border': torch.tensor([inner_list[2] for inner_list in concept_label]).cuda(),
                        'dermoscopic patterns': torch.tensor([inner_list[3] for inner_list in concept_label]).cuda(),
                        'texture': torch.tensor([inner_list[4] for inner_list in concept_label]).cuda(),
                        'symmetry': torch.tensor([inner_list[5] for inner_list in concept_label]).cuda(),
                        'elevation': torch.tensor([inner_list[6] for inner_list in concept_label]).cuda()
                    }


            loss_cls = self.cls_criterion(cls_logits, labels)
            #print(f"cls_logits shape: {cls_logits.shape}")
            #print(f"cls_with_te_logits shape: {cls_with_te_logits.shape}")
            if use_te_loss:
                loss_cls_with_te = self.cls_criterion(cls_with_te_logits, labels)
            doing_cnn_critical=False
            doing_cnn_trivial=False
            cnn_critical_loss_cls, cnn_trivial_loss_cls, cnn_logits_similarity_loss = [torch.tensor(0.0, device=images.device)]*3
            if doing_cnn_critical:
                cnn_critical_loss_cls = self.cls_criterion(cnn_logits_critical, labels)

            if doing_cnn_trivial:
                cnn_trivial_loss_cls = self.cls_criterion(cnn_logits_trivial, labels)

            cnn_loss = cnn_critical_loss_cls 
            #+ cnn_trivial_loss_cls
            if doing_cnn_critical:
                cnn_logits_similarity_loss = F.kl_div(F.softmax(cls_logits, dim=1).log(), F.softmax(cnn_logits_critical, dim=1), reduction='batchmean')
            #cnn_logits_dissimilarity_loss = -F.kl_div(F.softmax(cnn_logits_critical, dim=1).log(), F.softmax(cnn_logits_trivial, dim=1), reduction='batchmean')
            if self.do_logits_similarity:
                logits_similarity_loss = cnn_logits_similarity_loss
                #+ cnn_logits_dissimilarity_loss
            loss_concepts = 0
            idx = 0
            if self.concept_hardness=="hard":
                #print("hard concept hardness is used")
                for key in explicid.concept_token_dict.keys():
                    ###################################### Option simple CE
                    image_concept_loss = F.cross_entropy(image_logits_dict[key], concept_label[:, idx])
                    ###################################### Option weighted CE
                    #image_concept_loss = F.cross_entropy(image_logits_dict[key], concept_label[:, idx], weight=torch.FloatTensor(CONCEPT_CE_WEIGHTS[self.task][key]).cuda())
                    ######################################
                    loss_concepts += image_concept_loss
                    idx += 1
            elif self.concept_hardness=="soft_equal":
                for key in explicid.concept_token_dict.keys():
                    if key=="color" or key=="symmetry":
                        image_concept_loss = F.cross_entropy(image_logits_dict[key], concept_label[:, idx])
                    else:
                        predicted_probs = F.softmax(image_logits_dict[key], dim=1)
                        num_classes = image_logits_dict[key].shape[1]
                        one_hot_labels = torch.zeros((image_logits_dict[key].shape[0], num_classes), device=images.device)  # Initialize tensor of zeros
                        one_hot_labels.scatter_(1, concept_label[:, idx].unsqueeze(1), 1)  # Create one-hot encoding
                        soft_labels = label_smoothing(one_hot_labels, epsilon=0.1)
                        image_concept_loss = F.kl_div(torch.log(predicted_probs), soft_labels, reduction='batchmean')

                    ###############   Option use CE
                    #log_probs = F.log_softmax(image_logits_dict[key], dim=1)  # Compute log probabilities
                    #image_concept_loss = torch.sum(-soft_labels * log_probs, dim=1).mean()  # Average over batch
                    ###############
                    if image_concept_loss<0:
                        print(soft_labels)
                        print()
                        print(torch.log(predicted_probs))
                    loss_concepts += image_concept_loss
                    idx += 1
            elif self.concept_hardness=="soft_smarter":
                for key in explicid.concept_token_dict.keys():
                    predicted_probs = F.log_softmax(image_logits_dict[key], dim=1)  # Log probabilities for KL Divergence
                    loss_concepts += F.kl_div(predicted_probs,  concept_label_smooth_dict[key], reduction='batchmean')
                    # print(image_logits_dict[key])
                    # print()
                    # print(concept_label_smooth_dict[key])
                    #print(F.kl_div(predicted_probs,  concept_label_smooth_dict[key], reduction='none').mean(dim=-1).shape)
                    #print(F.kl_div(predicted_probs,  concept_label_smooth_dict[key], reduction='none'))
                    #print("=====")
                    #print(F.kl_div(predicted_probs,  concept_label_smooth_dict[key], reduction='none').mean(dim=-1))
                    #print(torch.FloatTensor(CE_WEIGHTS[self.task]).cuda()[labels])
                    #loss_concepts += (F.kl_div(predicted_probs,  concept_label_smooth_dict[key], reduction='none').mean(dim=-1)*torch.FloatTensor(CE_WEIGHTS[self.task]).cuda()[labels]).mean()*100  # Reduction is batchmean for proper scaling
                    idx += 1
            if epoch>=self.cls_loss_epoch:
                #################################### Option use loss_cls and loss_concepts
                if use_te_loss:
                    explicid_loss += loss_cls_with_te + loss_concepts / idx
                else:
                    explicid_loss += loss_cls + loss_concepts / idx
                #################################### Option use loss_cls only
                #explicid_loss += loss_cls
                #print(explicid_loss)
            elif epoch>10 and self.do_logits_similarity:
                explicid_loss += loss_cls + logits_similarity_loss+ loss_concepts / idx 
            else:
                explicid_loss += loss_concepts
                loss_cls_with_te*=0

            # if epoch>2:
            #     attn_explicd_loss+=overlap_loss/100
            
            # if epoch<8:
            #     logits_similarity_loss*=0
            #     #print("logits_similarity_loss ", logits_similarity_loss)
            #     cnn_loss_cls*=0
            if explicd_only==0 and do_sit:
                patches_output,model_output, model_input_pure_processed, zs_tilde, attn_sit_loss, sigmas_for_losses  = model(model_input, images, None,time_input.flatten(), **model_kwargs, 
                                                                        concept_label=None, 
                                                                        image_embeddings=agg_visual_tokens,
                                                                        cls_logits=None,
                                                                        attn_critical_weights=attn_critical_weights, 
                                                                        attn_trivial_weights=attn_trivial_weights,
                                                                        con_on_explicd_pred=True,
                                                                        vit_l_output=vit_l_output_input,
                                                                        longer_visual_tokens=longer_visual_tokens,
                                                                        critical_mask = critical_mask, 
                                                                        trivial_mask = trivial_mask,
                                                                        patchifyer_model=patchifyer_model,
                                                                        patches = patches)
                
                if imgs_indx==0:
                    #processing_loss = mean_flat((model_input_pure_processed - model_target) ** 2)
                    processing_loss  = torch.tensor(0.0, device=images.device)
                    if denoise_patches==0:
                        if self.noise_to_crit_map_only:
                            denoising_loss = mean_flat( ((model_output - model_target)*clipped_attention) ** 2)
                        else:
                            denoising_loss = mean_flat((model_output - model_target) ** 2)
                    elif denoise_patches==1:
                        denoising_loss = mean_flat((patches_output - patches) ** 2)
                    proj_loss_current = torch.tensor(0.0, device=images.device)
                    bsz = zs[0].shape[0]
                    for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                        for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                            z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                            z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                            proj_loss_current += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                    proj_loss_current /= (len(zs) * bsz)
                    proj_loss +=proj_loss_current

        loss_return_dict = {

            "explicid_loss":explicid_loss,
            "loss_cls":loss_cls,
            "logits_similarity_loss":logits_similarity_loss,

            "loss_cls_with_te":loss_cls_with_te,
            "te_loss":te_loss,

            "attn_explicd_loss":attn_explicd_loss,
            "sigmas_for_losses":sigmas_for_losses,
            "cnn_loss":cnn_loss,
            "overlap_loss":overlap_loss,
        }      
        if  explicd_only==0:
            loss_return_dict["processing_loss"]=processing_loss
            loss_return_dict["denoising_loss"]=denoising_loss
            loss_return_dict["proj_loss"]=proj_loss
            loss_return_dict["cosine_loss"]=cosine_loss
            loss_return_dict["sit_cls_loss"]=sit_cls_loss
            loss_return_dict["contr_loss"]=contr_loss
            loss_return_dict["loss_cls_criteria_only"]=loss_cls_criteria_only
            loss_return_dict["loss_cls_refined"]=loss_cls_refined
            loss_return_dict["attn_sit_loss"]=attn_sit_loss

        return loss_return_dict
        # if  explicd_only==0:
        #     return processing_loss, denoising_loss, proj_loss, explicid_loss, loss_cls, logits_similarity_loss, cosine_loss, sit_cls_loss, contr_loss, loss_cls_criteria_only, loss_cls_refined, torch.tensor(0.0, device=images.device), attn_explicd_loss, attn_map_loss_sit_total, sigmas_for_losses, cnn_loss
        # else:
        #     return None, None, None, explicid_loss, loss_cls, logits_similarity_loss, None, None, None, None, attn_explicd_loss, attn_sit_loss, sigmas_for_losses, cnn_loss

