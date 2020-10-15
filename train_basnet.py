import argparse
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import AUNet, AUNet_R16, DANet, R2UNet, SEUNet, UNetPP, UNet, RendDANet, BASNet

from data_loader import get_dataloader
from util import dice_coeff, get_accuracy, get_specificity, get_sensitivity, get_precision, get_F1
from loss import DiceLoss, MixLoss, BasLoss


def train_val(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader = get_dataloader(img_dir=config.train_img_dir, mask_dir=config.train_mask_dir, mode="train",
                                  batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = get_dataloader(img_dir=config.val_img_dir, mask_dir=config.val_mask_dir, mode="val",
                                batch_size=config.batch_size, num_workers=config.num_workers)

    writer = SummaryWriter(
        comment="LR_%f_BS_%d_MODEL_%s_DATA_%s" % (config.lr, config.batch_size, config.model_type, config.data_type))

    if config.model_type not in ['UNet', 'R2UNet', 'AUNet', 'R2AUNet', 'SEUNet', 'SEUNet++', 'UNet++',
                                 'DAUNet', 'DANet', 'AUNetR', 'RendDANet', "BASNet"]:
        print('ERROR!! model_type should be selected in supported models')
        print('Choose model %s' % config.model_type)
        return
    if config.model_type == "UNet":
        model = UNet()
    elif config.model_type == "AUNet":
        model = AUNet()
    elif config.model_type == "R2UNet":
        model = R2UNet()
    elif config.model_type == "SEUNet":
        model = SEUNet(useCSE=False, useSSE=False, useCSSE=True)
    elif config.model_type == "UNet++":
        model = UNetPP()
    elif config.model_type == "DANet":
        model = DANet(backbone='resnet101', nclass=1)
    elif config.model_type == "AUNetR":
        model = AUNet_R16(n_classes=1, learned_bilinear=True)
    elif config.model_type == "RendDANet":
        model = RendDANet(backbone='resnet101', nclass=1)
    elif config.model_type == "BASNet":
        model = BASNet(n_channels=3, n_classes=1)
    else:
        model = UNet()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device, dtype=torch.float)

    if config.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=config.lr, weight_decay=1e-6, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.loss == "dice":
        criterion = DiceLoss()
    elif config.loss == "bce":
        criterion = nn.BCELoss()
    elif config.loss == "bas":
        criterion = BasLoss()
    else:
        criterion = MixLoss()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    global_step = 0
    best_dice = 0.0
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        with tqdm(total=config.num_train, desc="Epoch %d / %d" % (epoch + 1, config.num_epochs),
                  unit='img') as train_pbar:
            model.train()
            for image, mask in train_loader:
                image = image.to(device, dtype=torch.float)
                mask = mask.to(device, dtype=torch.float)
                d0, d1, d2, d3, d4, d5, d6, d7 = model(image)
                loss = criterion(d0, d1, d2, d3, d4, d5, d6, d7, mask)
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)
                train_pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_pbar.update(image.shape[0])
                global_step += 1

                # if global_step % 100 == 0:
                #     writer.add_images('masks/true', mask, global_step)
                #     writer.add_images('masks/pred', d0 > 0.5, global_step)
            scheduler.step()
        epoch_dice = 0.0
        epoch_acc = 0.0
        epoch_sen = 0.0
        epoch_spe = 0.0
        epoch_pre = 0.0
        current_num = 0
        with tqdm(total=config.num_val, desc="Epoch %d / %d validation round" % (epoch + 1, config.num_epochs),
                  unit='img') as val_pbar:
            model.eval()
            locker = 0
            for image, mask in val_loader:
                current_num += image.shape[0]
                image = image.to(device, dtype=torch.float)
                mask = mask.to(device, dtype=torch.float)
                d0, d1, d2, d3, d4, d5, d6, d7 = model(image)
                batch_dice = dice_coeff(mask, d0).item()
                epoch_dice += batch_dice * image.shape[0]
                epoch_acc += get_accuracy(pred=d0, true=mask) * image.shape[0]
                epoch_sen += get_sensitivity(pred=d0, true=mask) * image.shape[0]
                epoch_spe += get_specificity(pred=d0, true=mask) * image.shape[0]
                epoch_pre += get_precision(pred=d0, true=mask) * image.shape[0]
                if locker == 200:
                    writer.add_images('masks/true', mask, epoch + 1)
                    writer.add_images('masks/pred', d0 > 0.5, epoch + 1)
                val_pbar.set_postfix(**{'dice (batch)': batch_dice})
                val_pbar.update(image.shape[0])
                locker += 1
            epoch_dice /= float(current_num)
            epoch_acc /= float(current_num)
            epoch_sen /= float(current_num)
            epoch_spe /= float(current_num)
            epoch_pre /= float(current_num)
            epoch_f1 = get_F1(SE=epoch_sen, PR=epoch_pre)
            if epoch_dice > best_dice:
                best_dice = epoch_dice
                writer.add_scalar('Best Dice/test', best_dice, epoch + 1)
                torch.save(model,
                           config.result_path + "/%s_%s_%d.pth" % (config.model_type, str(epoch_dice), epoch + 1))
            logging.info('Validation Dice Coeff: {}'.format(epoch_dice))
            print("epoch dice: " + str(epoch_dice))
            writer.add_scalar('Dice/test', epoch_dice, epoch + 1)
            writer.add_scalar('Acc/test', epoch_acc, epoch + 1)
            writer.add_scalar('Sen/test', epoch_sen, epoch + 1)
            writer.add_scalar('Spe/test', epoch_spe, epoch + 1)
            writer.add_scalar('Pre/test', epoch_pre, epoch + 1)
            writer.add_scalar('F1/test', epoch_f1, epoch + 1)

    writer.close()
    print("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=384)

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_type', type=str, default='BASNet', help='UNet/R2UNet/AUNet/SEUNet/UNet++/DANet/BASNet')
    parser.add_argument('--data_type', type=str, default='multi', help='single/multi')
    parser.add_argument('--loss', type=str, default='bas', help='bce/dice/mix')
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd/adam')

    parser.add_argument('--train_img_dir', type=str, default="../data/Multi_V1/train/image")
    parser.add_argument('--train_mask_dir', type=str, default="../data/Multi_V1/train/label")
    parser.add_argument('--val_img_dir', type=str, default="../data/Multi_V1/val/image")
    parser.add_argument('--val_mask_dir', type=str, default="../data/Multi_V1/val/label")
    parser.add_argument('--num_train', type=int, default=1600, help="4800/1600")
    parser.add_argument('--num_val', type=int, default=400, help="1200/400")
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--result_path', type=str, default='./exp')

    config = parser.parse_args()
    train_val(config)
