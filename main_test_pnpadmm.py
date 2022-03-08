import cv2
import csv
import os
import numpy as np
import time
import torch

from src.metrics import psnr, ssim
from src.config import get_cfg_defaults, get_args_from_parser
from src.models import make_model
from src.utils import mkdir, clear_result_dir
from src.data import tensor2complex, tensor2image, make_dataloader
from src.utils_pnp import load_models, recon_pnp_admm, get_denoiser

from scipy.io import savemat
import sys
sys.dont_write_bytecode = True

def main(cfg):
    dataset = make_dataloader(cfg.TEST, False)
    save_path = cfg.TEST.OUT_DIR
    mkdir(save_path)

    csv_path = os.path.join(save_path,'test_records.csv')
    with open(csv_path, 'w') as f:
        pass
    
    torch.manual_seed(1234)
    np.random.seed(1234)

    time_test = 0
    psnr_test = 0
    ssim_test = 0
    num_image = 0

    print(f'load weights from {cfg.TEST.WEIGHT_PATH}')
    models = load_models(cfg.TEST.WEIGHT_PATH, cfg)
    denoiser = lambda image_recon, sigma: get_denoiser(models, image_recon, sigma)
    for _, (file_path, batch) in enumerate(dataset): 
        image_target, image_zero_filled, kspace_sampled, mask = [i.cuda() for i in batch]
        mask = mask.permute(0,2,3,1)
        kspace_sampled = kspace_sampled.permute(0,2,3,1)

        time1 = time.time()
        image_predict, _ = recon_pnp_admm(denoiser, 
            image_zero_filled, 
            kspace_sampled, 
            mask,
            gamma=1.2,
            lam=6e-5,
            rho=0.0041,
            eta=1 + 1e-4,
            max_iter=100,
            eps=1e-6,
            verbose=False)
        time2 = time.time() - time1
        time_test += time2/image_target.shape[0]
        
        image_predict = tensor2complex(image_predict)
        image_zero_filled = tensor2image(image_zero_filled)
        image_target = tensor2image(image_target)

        for i, file_path_i in enumerate(file_path):
            num_image += 1
            file_name = file_path_i.split('/')[-1].split('.')[0]
            rec_im = image_predict[i]

            savemat(os.path.join(save_path,f'{file_name}.mat'), {'rec_im': rec_im})

            rec_im = np.clip(np.abs(rec_im), 0, 1)
            psnr_i = psnr(image_target[i], rec_im, data_range=1.)
            ssim_i = ssim(image_target[i], rec_im)

            cv2.imwrite(os.path.join(save_path,f'{file_name}.png'), 255*np.abs(rec_im))
            cv2.imwrite(os.path.join(save_path,f'{file_name}_zf.png'), 255*image_zero_filled[i])
            cv2.imwrite(os.path.join(save_path,f'{file_name}.png'), 255*np.abs(image_target[i]))

            psnr_test += psnr_i
            ssim_test += ssim_i
            print("File: %s Shape: %s, PSNR %f SSIM %.4f TIME %.4f" % (file_name, f'{rec_im.shape}',psnr_i, ssim_i, time2/image_target.shape[0]))

            with open(csv_path,'a') as f:
                writer = csv.writer(f)
                writer.writerow([file_name, psnr_i, ssim_i, time2/image_target.shape[0]])
    
    psnr_test /= num_image
    ssim_test /= num_image
    time_test /= len(dataset)
    print(f"PSNR: {psnr_test}, SSIM: {ssim_test}\n, TIME:{time_test}")
    with open(csv_path,'a') as f:
    	writer = csv.writer(f)
    	writer.writerow(['avr', psnr_test, ssim_test, time_test])

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    args = get_args_from_parser()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in cfg.SYSTEM.GPU_IDS[:cfg.SYSTEM.NUM_GPUS]])
    
    main(cfg)
