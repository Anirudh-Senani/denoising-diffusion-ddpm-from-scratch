"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    return torch.linspace(beta_start, beta_end, T)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1 - betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    at = torch.where(t<0, 1.0, a[t])
    return at[:, None, None, None]

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    bar_alpha_t = extract_into_batch(alphas_cumprod, t, x0)
    xt = torch.sqrt(bar_alpha_t) * x0 + torch.sqrt(1 - bar_alpha_t) * noise

    return xt

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = alphas_from_betas(betas)

    alphas_cumprod = cumprod_alphas(alphas)

    return dict(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1-alphas_cumprod),
        T=T
    )

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    return ((noise - noise_pred)**2).mean()

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    xt = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(xt, t)
    loss = noise_prediction_loss(noise_pred, noise)

    return loss

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    half = dim//2
    if half == 1:
        pos = torch.tensor([1/10000])
    else:
        inds = torch.arange(half)/(half-1)
        pos = 1/(10000**inds)
    
    rads = torch.outer(t, pos)
    out = torch.zeros((t.shape[0], dim))

    out[:, :half] = torch.sin(rads)
    out[:, half:] = torch.cos(rads)
    return out

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    # TODO: initialize tiny residual denoiser parameters
    torch.manual_seed(seed)
    std = 0.02
    conv_in_w = torch.normal(mean=0, std=std, size=(hidden, in_ch, 3, 3), requires_grad=True)
    conv_in_b = torch.zeros(hidden, requires_grad=True)
    time_mlp_w = torch.normal(mean=0, std=std, size=(hidden, time_dim), requires_grad=True)
    time_mlp_b = torch.zeros(hidden, requires_grad=True)
    conv_mid_w = torch.normal(mean=0, std=std, size=(hidden, hidden, 3, 3), requires_grad=True)
    conv_mid_b = torch.zeros(hidden, requires_grad=True)
    conv_out_w = torch.normal(mean=0, std=std, size=(in_ch, hidden, 3, 3), requires_grad=True)
    conv_out_b = torch.zeros(in_ch, requires_grad=True)

    return dict(
        conv_in_w=conv_in_w,
        conv_in_b=conv_in_b,
        time_mlp_w=time_mlp_w,
        time_mlp_b=time_mlp_b,
        conv_mid_w=conv_mid_w,
        conv_mid_b=conv_mid_b,
        conv_out_w=conv_out_w,
        conv_out_b=conv_out_b
    )

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # TODO: time-conditioned tiny CNN predicting noise
    h = F.conv2d(x, params['conv_in_w'], bias=params['conv_in_b'], padding=1)
    temb = timestep_embedding(t, params['time_mlp_w'].shape[1])
    temb = F.relu(F.linear(temb, params['time_mlp_w'], bias=params['time_mlp_b']))

    h += temb[:, :, None, None]
    h = F.relu(h)
    h = F.relu(F.conv2d(h, params['conv_mid_w'], bias=params['conv_mid_b'], padding=1))

    return F.conv2d(h, params['conv_out_w'], bias=params['conv_out_b'], padding=1)

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    # TODO: n images with a random bright disk on a black background
    torch.manual_seed(seed)
    radius = size//4
    targets = torch.randint(radius, size-radius, (n, 2), dtype=torch.float32)

    grid_range = torch.arange(size, dtype=torch.float32)

    target_r = targets[:, 0].view(n, 1, 1)
    target_c = targets[:, 1].view(n, 1, 1)

    dist = (grid_range.view(1, size, 1) - target_r)**2 + (grid_range.view(1, 1, size) - target_c)**2

    return torch.where(dist>radius, 0.0, 1.0).view(n, 1, size, size)

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    # TODO: sample t,noise -> loss -> SGD on params
    torch.manual_seed(seed)

    t = torch.randint(high=schedule['T'], size=(1,))
    noise = torch.normal(0.0, 1.0, size=x0.shape)
    model = lambda x, t: tiny_unet_forward(x, t, params)
    loss = diffusion_training_loss(model, x0, t, noise, schedule['alphas_cumprod'])
    loss.backward()

    new_params = {}
    for p in params:
        if params[p].grad is not None:
            new_params[p] = (params[p] - lr*params[p].grad).detach().requires_grad_(True)
        else:
            new_params[p] = params[p].clone()

    return new_params, loss.detach().item()

# Step 14 - train_ddpm
import torch
import torch.nn.functional as F

def train_ddpm(dataset, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:
    # TODO: minibatch SGD training loop
    history = []
    for step in range(num_steps):
        inds = torch.randint(0, dataset.shape[0], (batch_size,))
        batch = dataset[inds]
        params, loss = ddpm_train_step(params, batch, schedule, lr, seed+step)

        history.append(loss)

    return params, history

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # TODO: invert the q_sample equation for x0
    alpha_bar_t = extract_into_batch(alphas_cumprod, t, x_t)
    x0_hat = (x_t - torch.sqrt(1 - alpha_bar_t)*eps)/torch.sqrt(alpha_bar_t)

    return x0_hat

# Step 16 - ddpm_p_mean_variance
import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    # TODO: return (posterior_mean, variance, x0_hat)
    x0_hat = predict_x0_from_eps(x_t, t, eps, schedule['alphas_cumprod'])
    x0_hat = torch.clamp(x0_hat, -1, 1)
    beta_t = extract_into_batch(schedule['betas'], t, x0_hat)
    sqrt_alpha_bar_t_1 = extract_into_batch(schedule['sqrt_alphas_cumprod'], t-1, x0_hat)
    one_minus_alpha_bar_t = extract_into_batch(1.0-schedule['alphas_cumprod'], t, x0_hat)
    sqrt_alpha_bar_t = extract_into_batch(schedule['sqrt_alphas_cumprod'], t, x0_hat)
    one_minus_alpha_bar_t_1 = extract_into_batch(1.0-schedule['alphas_cumprod'], t-1, x0_hat)

    mu = (sqrt_alpha_bar_t_1 * beta_t/one_minus_alpha_bar_t) * x0_hat + (sqrt_alpha_bar_t * one_minus_alpha_bar_t_1/one_minus_alpha_bar_t)*x_t
    sigma = beta_t

    return mu, sigma, x0_hat

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

