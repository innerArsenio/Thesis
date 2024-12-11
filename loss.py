import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from sklearn.metrics import balanced_accuracy_score
CONCEPT_LABEL_MAP = [
            [3, 0, 0, 3, 3, 0, 2], # AKIEC
            [2, 0, 2, 2, 2, 0, 1], # BCC
            [4, 2, 1, 4, 4, 1, 3], # BKL
            [5, 1, 1, 5, 5, 1, 0], # DF
            [0, 0, 0, 0, 0, 0, 0], # MEL
            [1, 1, 1, 1, 1, 1, 0], # NV
            [6, 3, 1, 6, 1, 2, 0], # VASC
        ]

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
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.lesion_weight = torch.FloatTensor([ 0.134, 0.084, 0.039, 0.389, 0.039, 0.0065, 0.305]).cuda()
        self.cls_criterion = nn.CrossEntropyLoss(weight=self.lesion_weight).cuda()

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

    def __call__(self, model, images, model_kwargs=None, zs=None, labels=None, explicid=None, explicid_imgs_list= None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
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
        model_input = alpha_t * images + sigma_t * noises
        #print(f"model_input shape: {model_input.shape}")
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        denoising_loss, proj_loss, explicid_loss, cosine_loss, sit_cls_loss = 0, 0, 0, 0, 0
        contr_loss= torch.tensor(0.0, device=images.device)
        list_of_emb_concepts= []
        for imgs_indx, explicid_imgs in enumerate(explicid_imgs_list):
            cls_logits, image_logits_dict, agg_visual_tokens = explicid(explicid_imgs)
            concept_label = torch.tensor([CONCEPT_LABEL_MAP[label] for label in labels])
            concept_label = concept_label.long().cuda()
            loss_cls = self.cls_criterion(cls_logits, labels)
            #print(f"cls logist shape: {cls_logits.shape}")
            #print(f"labels shape: {labels.shape}")
            loss_concepts = 0
            idx = 0
            for key in explicid.concept_token_dict.keys():
                image_concept_loss = F.cross_entropy(image_logits_dict[key], concept_label[:, idx])
                loss_concepts += image_concept_loss
                idx += 1

            explicid_loss += loss_cls + loss_concepts / idx
            model_output, image_embed, embedded_concepts, produced_concepts, y_predicted, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs, 
                                                                    concept_label=concept_label, 
                                                                    image_embeddings=agg_visual_tokens)
            list_of_emb_concepts.append(embedded_concepts)
            #print(f"y_predicted shape: {y_predicted.shape}")
            sit_cls_loss = self.cls_criterion(y_predicted, labels)

            if imgs_indx==0:
                denoising_loss = mean_flat((model_output - model_target) ** 2)
                # projection loss
                proj_loss = 0.
                bsz = zs[0].shape[0]
                for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                    for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                        z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                        z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                        proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                proj_loss /= (len(zs) * bsz)

            #print(f"produced_concepts shape: {produced_concepts.shape}")
            cosine_sim = F.cosine_similarity(embedded_concepts, produced_concepts, dim=-1)
            cosine_loss += 1 - cosine_sim.mean()

            # mse_criterion = nn.MSELoss()

            # # Mean squared error loss between the model's output vector and the label's embedding
            # mse_loss = mse_criterion(embedded_concepts, produced_concepts)*200
        if len(list_of_emb_concepts)>1:
            contr_loss+=contrastive_loss(list_of_emb_concepts[0], list_of_emb_concepts[1], positive=True)
            contr_loss+=contrastive_loss(list_of_emb_concepts[0], list_of_emb_concepts[2], positive=True)
            contr_loss+=contrastive_loss(list_of_emb_concepts[0], list_of_emb_concepts[3], positive=True)
            contr_loss+=contrastive_loss(list_of_emb_concepts[0], list_of_emb_concepts[4], positive=True)

        return denoising_loss, proj_loss, explicid_loss, loss_cls, cosine_loss, sit_cls_loss, contr_loss
