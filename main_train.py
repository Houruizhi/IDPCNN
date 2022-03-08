import os
import torch
from tqdm import tqdm
import cv2
import shutil
import tensorboardX

from src.config import get_cfg_defaults, get_args_from_parser
from src.modules import make_module
from src.utils import mkdir, initial_seed
from src.metrics import batch_PSNR, psnr
from src.data import make_dataloader, tensor2image
from src.utils import save_state_dict, load_model, clear_result_dir

def main(cfg):
    print(cfg)
    # initialize the training
    MIDDLE_RESULT_DIR = os.path.join(cfg.TRAIN.OUT_DIR, 'middle_res')
    mkdir(cfg.TRAIN.OUT_DIR)
    mkdir(MIDDLE_RESULT_DIR)
    log_file_path = os.path.join(cfg.TRAIN.OUT_DIR, 'train_log.txt')
    
    print(f'log dir: {cfg.TRAIN.OUT_DIR}')
    with open(log_file_path, 'w') as f:
        pass
    summary_path = os.path.join(cfg.TRAIN.OUT_DIR, 'tensorboard')
    
    if not cfg.TRAIN.RESUME:
        if os.path.exists(summary_path):
            shutil.rmtree(summary_path, ignore_errors=True)

    tensorboard_logger = tensorboardX.SummaryWriter(log_dir=summary_path)

    # make the dataloader
    loader_train = make_dataloader(cfg.DATASET.TRAIN, train=True)
    loader_val = make_dataloader(cfg.DATASET.VAL, train=False)

    # make the model and optimizer
    module = make_module(cfg)
        
    optimizer = torch.optim.Adam(module.model.parameters(), lr=cfg.SOLVER.LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
        list(range(cfg.SOLVER.MILESTONE, cfg.TRAIN.NUM_EPOCHS, cfg.SOLVER.MILESTONE)), gamma = cfg.SOLVER.LEARNING_RATE_DECAY)

    epoch_start = 0
    if cfg.TRAIN.RESUME:
        chp = torch.load(os.path.join(cfg.TRAIN.OUT_DIR, 'net.pth'))
        module.model = load_model(module.model, chp['weights'])
        for _ in range(chp['epoch']):
            optimizer.step()
            scheduler.step()
        epoch_start = chp['epoch']+1
        optimizer.load_state_dict(chp['optimizer'])

    step = 1
    for epoch in range(epoch_start, cfg.TRAIN.NUM_EPOCHS):   
        loss_record = 0
        psnr_record = 0
        with tqdm(total=len(loader_train), desc='Epoch: [%d/%d], lr: [%.6f]'%\
            (epoch, cfg.TRAIN.NUM_EPOCHS, optimizer.param_groups[0]["lr"]), miniters=1) as t:
            for file_name, batch in loader_train:
                module.model.train()
                module.model.zero_grad()
                optimizer.zero_grad()

                loss, image_target, image_predict, image_noisy = module.train_step(batch)
                
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    psnr_train = batch_PSNR(image_target, image_predict)
                tensorboard_logger.add_scalar('loss', loss.item(), global_step=step)
                loss_record += loss.item()
                psnr_record += psnr_train
                t.set_postfix_str("Batch Loss: %.4f, Batch PSNR: %.2f" % (loss.item(), psnr_train))
                t.update()
                step += 1
                
            if cfg.TRAIN.SHOW_MIDDLE_RESULTS:
                Img = tensor2image(image_predict).clip(0., 1.)
                Inoisy  = tensor2image(image_noisy).clip(0., 1.)
                Iclean = tensor2image(image_target).clip(0., 1.)

                for i in range(Img.shape[0]):
                    cv2.imwrite(os.path.join(MIDDLE_RESULT_DIR, f'{epoch}_{i}_clean.png'), 255*Iclean[i])
                    cv2.imwrite(os.path.join(MIDDLE_RESULT_DIR, f'{epoch}_{i}_noisy.png'), 255*Inoisy[i])
                    cv2.imwrite(os.path.join(MIDDLE_RESULT_DIR, f'{epoch}_{i}_out.png'), 255*Img[i])

        scheduler.step()

        checkpoint_path = os.path.join(cfg.TRAIN.OUT_DIR, f'net.pth')
        chp = {
            'epoch': epoch, 
            'weights': module.model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
        torch.save(chp, checkpoint_path)

        loss_record /= len(loader_train)
        psnr_record /= len(loader_train)
        log_text = f"[epoch {epoch}] average train loss: {loss_record}, train PSNR: {psnr_record}"

        if loader_val:
            module.model.eval()
            psnr_val = 0
            for file_name, batch in loader_val:
                with torch.no_grad():
                    image_target, image_predict, image_noisy = module.val_step(batch)

                psnr_val += batch_PSNR(image_target, image_predict)
            
            tensorboard_logger.add_scalar('val_psnr_avg', psnr_val/len(loader_val), global_step=epoch)
            
            log_text = log_text + f", val PSNR:{psnr_val/len(loader_val)}"

        log_text += '\n'
        print(log_text)
        with open(log_file_path, 'a') as f:
            f.writelines([log_text])

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    args = get_args_from_parser()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in cfg.SYSTEM.GPU_IDS[:cfg.SYSTEM.NUM_GPUS]])
    
    # main(cfg)
    
    outf = cfg.TRAIN.OUT_DIR
    # noise_levels = [i/10 for i in range(1,10)]+list(range(1,26))
    # noise_levels = [i/10 for i in range(1,10)]
    noise_levels = list(range(1,13))
    # noise_levels = list(range(13,26))
    noise_levels = noise_levels[::-1]
    for noisel in noise_levels:
        cfg.DATASET.TRAIN.NOISE_LEVEL = float(noisel)
        cfg.DATASET.VAL.NOISE_LEVEL = float(noisel)
        cfg.TRAIN.OUT_DIR = outf + '_' + str(int(cfg.DATASET.TRAIN.NOISE_LEVEL*10)/10)
        print(cfg)
        main(cfg)