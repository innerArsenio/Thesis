import torch
import numpy as np

CONCEPT_LABEL_MAP = [
            [3, 0, 0, 3, 3, 0, 2], # AKIEC
            [2, 0, 2, 2, 2, 0, 1], # BCC
            [4, 2, 1, 4, 4, 1, 3], # BKL
            [5, 1, 1, 5, 5, 1, 0], # DF
            [0, 0, 0, 0, 0, 0, 0], # MEL
            [1, 1, 1, 1, 1, 1, 0], # NV
            [6, 3, 1, 6, 1, 2, 0], # VASC
        ]

def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur

def interpolant(t):
    alpha_t = 1 - t
    sigma_t = t
    d_alpha_t = -1
    d_sigma_t =  1

    return alpha_t, sigma_t, d_alpha_t, d_sigma_t

def euler_sampler(
        model,
        latents,
        y,
        visual_tokens,
        cls_logits,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear", # not used, just for compatability
        attn_critical_weights=None, 
        attn_trivial_weights=None,
        longer_visual_tokens = None,
        vit_l_output=None,
        critical_mask=None, 
        trivial_mask=None,
        highlight_the_critical_mask=False,
        use_actual_latent_of_the_images=1
        ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([6] * y.size(0), device=y.device)
        #imgs_null= torch.zeros_like(visual_tokens)
        cls_logits_null = torch.zeros_like(cls_logits)
    _dtype = latents.dtype    
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    #########################################################
    x_next = latents.to(torch.float64)
    vit_l_output_next = vit_l_output
    #########################################################
    #time_input = torch.rand((y.shape[0], 1, 1, 1))
    #time_input = time_input.to(device=y.device, dtype=latents.dtype)
    #alpha_t, sigma_t, _, _ = interpolant(time_input)
    #x_next = alpha_t * latents + sigma_t * torch.randn_like(latents)
    #########################################################s
    device = x_next.device
    concept_label = torch.tensor([CONCEPT_LABEL_MAP[label] for label in y])
    concept_label = concept_label.long().cuda()
    
    with torch.no_grad():
        # samples=patchifyer_model.patchify_the_latent(latents)
        # samples=patchifyer_model.unpatchify_the_latent(samples)
        # return samples
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            vit_l_output_cur = vit_l_output_next
            #print("x next")
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
                imgs_tokens_input = torch.cat([visual_tokens]*2, dim=0)
                concept_label_input = torch.cat([concept_label]*2, dim=0)
                cls_logits_input = torch.cat([cls_logits, cls_logits_null], dim=0)
                attn_critical_weights_input =  torch.cat([attn_critical_weights]*2, dim=0)
                attn_trivial_weights_input =  torch.cat([attn_trivial_weights]*2, dim=0)
            else:
                model_input = x_cur
                y_cur = y
                vit_l_output = vit_l_output_cur
                imgs_tokens_input = visual_tokens
                concept_label_input = concept_label
                cls_logits_input = cls_logits
                attn_critical_weights_input =  attn_critical_weights
                attn_trivial_weights_input =  attn_trivial_weights 
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            #print("1")
            #text_embed=torch.randn(256, 512).to(device)
            #img_embed=torch.randn(256, 512).to(device)

            patches_output, d_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs,  # if need be will fill in model_pure and model_target
                concept_label=concept_label_input, 
                image_embeddings=imgs_tokens_input,
                cls_logits=cls_logits_input,
                attn_critical_weights=attn_critical_weights_input, 
                attn_trivial_weights=attn_trivial_weights_input,
                longer_visual_tokens=longer_visual_tokens,
                vit_l_output=vit_l_output,
                critical_mask= critical_mask, 
                trivial_mask = trivial_mask,
                highlight_the_critical_mask=highlight_the_critical_mask
                )
            
            if use_actual_latent_of_the_images==1:
                return patches_output, d_cur.to(torch.float32)
            #.to(torch.float64)
            #patches_output = patches_output.to(torch.float64)
            # d_cur = d_cur.to(torch.float64)
            # d_cur_pure = d_cur_pure.to(torch.float64)
            # d_cur_critical_removed = d_cur_critical_removed.to(torch.float64)
            
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)                
            x_next = x_cur + (t_next - t_cur) * d_cur
            #vit_l_output_next = vit_l_output_cur + (t_next - t_cur) * x_vit_l_output
            #x_next = d_cur
            
            #print("2")
            if heun and (i < num_steps - 1):
                #print("3")
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    model_input = torch.cat([x_next] * 2)
                    y_cur = torch.cat([y, y_null], dim=0)
                else:
                    model_input = x_next
                    y_cur = y
                kwargs = dict(y=y_cur)
                time_input = torch.ones(model_input.size(0)).to(
                    device=model_input.device, dtype=torch.float64
                    ) * t_next
                d_prime = model(
                    model_input.to(dtype=_dtype), latents, None,  time_input.to(dtype=_dtype), **kwargs,
                    concept_label=concept_label_input, 
                    image_embeddings=imgs_tokens_input,
                    cls_logits=cls_logits_input,
                    attn_critical_weights=attn_critical_weights_input, 
                    attn_trivial_weights=attn_trivial_weights_input,
                    longer_visual_tokens=longer_visual_tokens,
                    vit_l_output=vit_l_output,
                    critical_mask= critical_mask, 
                    trivial_mask = trivial_mask
                    )[0].to(torch.float64)
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                    d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
                
    #return x_next
    return patches_output, x_next.to(torch.float32)


def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        ):
    # setup conditioning
    if cfg_scale > 1.0:
        # 1000 is the number of classes here
        y_null = torch.tensor([7] * y.size(0), device=y.device)
            
    _dtype = latents.dtype
    
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y            
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)            
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                )[0].to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
    
    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y            
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    v_cur = model(
        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
        )[0].to(torch.float64)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
                    
    return mean_x
