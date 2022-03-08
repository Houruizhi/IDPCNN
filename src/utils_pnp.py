
import os
import time
import torch
import numpy as np
from .models import make_model
from .metrics import batch_PSNR
from .fft import fft2c, ifft2c
from .data import tensor_split

def cuda2np(img):
    return img.squeeze().cpu().numpy()

def projection_tensor(image_denoised,data_kspace,mask,rho = 2e-2):
    '''
    input shape: (n,h,w,2)
    output shape: (n,h,w,2)
    '''
    image_projection = rho*fft2c(image_denoised) + data_kspace*mask
    return ifft2c(image_projection/(rho+mask))

def load_models(pretrained_path, cfg):
    nets_path = os.listdir(pretrained_path)
    models = {}
    for file_name in nets_path:
        model = make_model(cfg.MODEL)
        noise_level = file_name.split('_')[-1]
        noise_level = float(noise_level)
        if noise_level < 1:
            noise_level = '%.1f'%(noise_level)
        else:
            noise_level = str(int(noise_level))
        model_path = os.path.join(pretrained_path, file_name, 'net.pth')
        model.load_state_dict(torch.load(os.path.join(model_path))['weights'])
        models[str(noise_level)] = model.cuda()
    return models

def get_denoiser(models, image_recon, sigma):
    model_keys = list(models.keys())
    sigmas = np.array([float(i) for i in list(models.keys())])/255.
    dis = np.abs(sigmas - sigma)
    sigma = model_keys[dis.argmin()]
    return models[str(sigma)](image_recon)

def recon_pnp_admm(
                    denoiser,
                    image_recon, 
                    data_kspace, 
                    mask, 
                    image_target=None,
                    gamma=1.2,
                    lam=6e-5,
                    rho=0.0041,
                    eta=1 + 1e-4,
                    max_iter=100,
                    eps=1e-6,
                    verbose=False
                    ):

    psnrs = []
    deltas = []
    
    rho_k = rho
    delta = 0
    count_iter = 0

    if image_target is not None:
        PSNR = batch_PSNR(image_target, image_recon)
        psnrs.append(PSNR)
    
    time1 = time.time()
    for _ in range(max_iter):
        delta_old = delta
        image_recon_old = image_recon.clone()

        sigma = np.sqrt(lam/rho_k)

        with torch.no_grad():
            image_recon = denoiser(image_recon, sigma)

        image_recon = image_recon.permute(0,2,3,1)
        image_recon = projection_tensor(image_recon,data_kspace,mask,rho=1e-3*rho_k)
        image_recon = image_recon.permute(0,3,1,2)

        delta = torch.mean((image_recon_old-image_recon)**2)
        deltas.append(delta)
        if delta > eta*delta_old:
            rho_k = gamma*rho_k
        if delta < eps:
            break
        if image_target is not None:
            PSNR = batch_PSNR(image_target, image_recon)
            psnrs.append(PSNR)
        count_iter += 1 
        if verbose:
            print(f'iter: {count_iter}, sigma: {int(sigma*255)}, PSNR: {PSNR}')
        
    TIME = time.time()-time1
    return image_recon, (psnrs, deltas, TIME)