import cv2
import csv
import os
import math
import argparse
import numpy as np
import time
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from src.metrics import psnr, ssim
from src.config import get_cfg_defaults, get_args_from_parser
from src.models import make_model
from src.modules import make_module
from src.utils import mkdir, clear_result_dir
from src.data import tensor2image, make_dataloader
from src.utils import load_model

import sys
sys.dont_write_bytecode = True

def main(cfg):
    model = make_model(cfg.MODEL)
    dataset = make_dataloader(cfg.TEST, False)
    save_path = cfg.TEST.OUT_DIR
    os.makedirs(save_path, exist_ok=True)

    weights_path = os.path.join(cfg.TEST.WEIGHT_PATH, )
    print(f'load weights from {weights_path}')
    weights = torch.load(weights_path)
    module = make_module(cfg)
    module.model = load_model(module.model, weights['weights'])

    clear_result_dir(save_path)

    mkdir(save_path)
    csv_path = os.path.join(save_path,'test_records.csv')
    with open(csv_path, 'w') as f:
        pass

    torch.manual_seed(1234)
    np.random.seed(1234)

    time_test = 0
    psnr_test = 0
    ssim_test = 0

    for i, (file_path, batch) in enumerate(dataset): 
        file_name = file_path[0].split('/')[-1].split('.')[0]

        time1 = time.time()
        with torch.no_grad():
            image_target, image_predict, image_noisy = module.val_step(batch)
        time2 = time.time() - time1
        
        out = tensor2image(image_predict).squeeze().clip(0.,1.)
        image_noisy = tensor2image(image_noisy).squeeze().clip(0.,1.)
        image_target = tensor2image(image_target).squeeze()

        psnr_i = psnr(image_target, out, data_range=1.)
        ssim_i = ssim(image_target, out) 
        psnr_test += psnr_i
        ssim_test += ssim_i
        time_test += time2

        print("File: %s Shape: %s, PSNR %f SSIM %.4f TIME %.4f" % (file_name, f'{image_target.shape}',psnr_i, ssim_i, time2))
        
        cv2.imwrite('%s/%s_%.4f.png'%(save_path, file_name, psnr_i), out * 255)
        cv2.imwrite(f'{save_path}/{file_name}_noisy.png', image_noisy * 255)
        cv2.imwrite(f'{save_path}/{file_name}.png', image_target * 255)

        with open(csv_path,'a') as f:
        	writer = csv.writer(f)
        	writer.writerow([i, psnr_i, ssim_i, time2])
    
    psnr_test /= len(dataset)
    ssim_test /= len(dataset)
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
