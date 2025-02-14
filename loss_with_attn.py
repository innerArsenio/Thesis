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

    'IDRID': [ 0.076, 0.506, 0.074, 0.137, 0.207],
    'IDRID_EDEMA': [0.1606, 0.6935, 0.1458],

    'BUSI': [ 0.157, 0.327, 0.516],
    'BUSI_SOFT': [ 0.157, 0.327, 0.516]
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
            concept_hardness="soft_equal"
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

    def __call__(self, model, images, latent_raw_image, model_kwargs=None, zs=None, labels=None, explicid=None, explicid_imgs_list= None, epoch=None,  explicd_only=0, do_sit=False, do_pretraining_the_patchifyer=False, patchifyer_model=None):
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
        model_input = images
        if self.prediction == 'v':
            model_target = images
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        denoising_loss, proj_loss, explicid_loss, cosine_loss, sit_cls_loss, logits_similarity_loss, attn_map_loss_sit_total = [torch.tensor(0.0, device=images.device)]*7
        contr_loss= torch.tensor(0.0, device=images.device)
        attn_explicd_loss=torch.tensor(0.0, device=images.device)
        loss_cls_criteria_only = torch.tensor(0.0, device=images.device)
        loss_cls_refined = torch.tensor(0.0, device=images.device)
        sit_cls_loss = torch.tensor(0.0, device=images.device)
        sigmas_for_losses = {}
        for imgs_indx, explicid_imgs in enumerate(explicid_imgs_list):
            if do_pretraining_the_patchifyer:
                model_output = model(model_input)
                processing_loss = mean_flat((model_output - model_target) ** 2)
                return processing_loss, [torch.tensor(0.0, device=images.device)]*15
            with torch.no_grad():
                explicid_imgs_latents=patchifyer_model.patchify_the_latent(latent_raw_image)
            patches, patches_colored, cls_logits, cls_minimal_logits, _, image_logits_dict, agg_visual_tokens, agg_trivial_tokens, attn_explicd_loss, overlap_loss, attn_critical_weights, attn_trivial_weights, vit_l_output, agg_critical_visual_tokens, agg_trivial_visual_tokens, cnn_logits_critical, cnn_logits_trivial, critical_mask, trivial_mask  = explicid(explicid_imgs, explicid_imgs_latents=explicid_imgs_latents)
            if explicd_only==0:
                agg_visual_tokens = torch.cat((agg_visual_tokens, agg_trivial_tokens), dim=1)
                longer_visual_tokens = torch.cat((agg_critical_visual_tokens, agg_trivial_visual_tokens), dim=1)
                vit_l_output_input = vit_l_output


            if self.concept_hardness!="soft_smarter":
                concept_label = torch.tensor([self.concept_label_map[label] for label in labels])
                concept_label = concept_label.long().cuda()
            else:
                concept_label = [self.concept_label_map[label] for label in labels]
                concept_label_busi_soft_smooth_dict = {
                    'shape': torch.tensor([inner_list[0] for inner_list in concept_label]).cuda(),
                    'margin': torch.tensor([inner_list[1] for inner_list in concept_label]).cuda(),
                    'echo pattern': torch.tensor([inner_list[2] for inner_list in concept_label]).cuda(),
                    'posterior features': torch.tensor([inner_list[3] for inner_list in concept_label]).cuda(),
                    'calcifications': torch.tensor([inner_list[4] for inner_list in concept_label]).cuda(),
                    'orientation': torch.tensor([inner_list[5] for inner_list in concept_label]).cuda()

                }

            loss_cls = self.cls_criterion(cls_minimal_logits, labels)
            cnn_critical_loss_cls = self.cls_criterion(cnn_logits_critical, labels)
            cnn_trivial_loss_cls = self.cls_criterion(cnn_logits_trivial, labels)
            cnn_loss = cnn_critical_loss_cls 
            #+ cnn_trivial_loss_cls
            cnn_logits_similarity_loss = F.kl_div(F.softmax(cls_minimal_logits, dim=1).log(), F.softmax(cnn_logits_critical, dim=1), reduction='batchmean')
            #cnn_logits_dissimilarity_loss = -F.kl_div(F.softmax(cnn_logits_critical, dim=1).log(), F.softmax(cnn_logits_trivial, dim=1), reduction='batchmean')
            if self.do_logits_similarity:
                logits_similarity_loss = cnn_logits_similarity_loss
                #+ cnn_logits_dissimilarity_loss
            loss_concepts = 0
            idx = 0
            if self.concept_hardness=="hard":
                #print("hard concept hardness is used")
                for key in explicid.concept_token_dict.keys():
                    image_concept_loss = F.cross_entropy(image_logits_dict[key], concept_label[:, idx])
                    loss_concepts += image_concept_loss
                    idx += 1
            elif self.concept_hardness=="soft_equal":
                for key in explicid.concept_token_dict.keys():
                    predicted_probs = F.softmax(image_logits_dict[key], dim=1)
                    num_classes = image_logits_dict[key].shape[1]
                    one_hot_labels = torch.zeros((image_logits_dict[key].shape[0], num_classes), device=images.device)  # Initialize tensor of zeros
                    one_hot_labels.scatter_(1, concept_label[:, idx].unsqueeze(1), 1)  # Create one-hot encoding
                    soft_labels = label_smoothing(one_hot_labels, epsilon=0.1)
                    image_concept_loss = F.kl_div(torch.log(predicted_probs), soft_labels, reduction='batchmean')
                    loss_concepts += image_concept_loss
                    idx += 1
            elif self.concept_hardness=="soft_smarter":
                for key in explicid.concept_token_dict.keys():
                    predicted_probs = F.log_softmax(image_logits_dict[key], dim=1)  # Log probabilities for KL Divergence
                    loss_concepts += F.kl_div(predicted_probs,  concept_label_busi_soft_smooth_dict[key], reduction='batchmean')  # Reduction is batchmean for proper scaling
                    idx += 1

            if epoch>=0:
                explicid_loss += loss_cls + loss_concepts / idx
                #print(explicid_loss)
            elif epoch>10 and self.do_logits_similarity:
                explicid_loss += loss_cls + logits_similarity_loss+ loss_concepts / idx 
            else:
                explicid_loss += loss_concepts

            # if epoch>2:
            #     attn_explicd_loss+=overlap_loss/100
            
            # if epoch<8:
            #     logits_similarity_loss*=0
            #     #print("logits_similarity_loss ", logits_similarity_loss)
            #     cnn_loss_cls*=0

            if explicd_only==0 and do_sit:
                patches_output,model_output, model_input_pure_processed, zs_tilde, attn_map_loss_sit_total, sigmas_for_losses  = model(model_input, images, None,time_input.flatten(), **model_kwargs, 
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
                                                                        patchifyer_model=patchifyer_model)
                
                if imgs_indx==0:
                    #processing_loss = mean_flat((model_input_pure_processed - model_target) ** 2)
                    processing_loss  = torch.tensor(0.0, device=images.device)
                    #denoising_loss = mean_flat((model_output - model_target) ** 2)
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

            
        if  explicd_only==0:
            return processing_loss, denoising_loss, proj_loss, explicid_loss, loss_cls, logits_similarity_loss, cosine_loss, sit_cls_loss, contr_loss, loss_cls_criteria_only, loss_cls_refined, torch.tensor(0.0, device=images.device), attn_explicd_loss, attn_map_loss_sit_total, sigmas_for_losses, cnn_loss
        else:
            return None, None, None, explicid_loss, loss_cls, logits_similarity_loss, None, None, None, None, attn_explicd_loss, attn_map_loss_sit_total, sigmas_for_losses, cnn_loss

