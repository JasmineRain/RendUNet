import torch
import os
from data_loader import get_dataloader
from util import dice_coeff
import numpy as np
from PIL import Image
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def inference():

    img_dir = "../data/Multi_V1/val/image"
    mask_dir = "../data/Multi_V1/val/label"
    out_dir = "./visualization"
    val_loader = get_dataloader(img_dir=img_dir, mask_dir=mask_dir, mode="test",
                                batch_size=1, num_workers=4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_dir = "./exp"

    model_names = os.listdir(model_dir)

    for model_name in model_names:

        # print(os.path.join(model_dir, model_name))
        name = model_name.split('_')[0]
        if name not in ['RendUNet']:
            continue

        model = torch.load(os.path.join(model_dir, model_name))
        model = model.to(device)
        model.eval()
        print(name)
        seq = 1
        for image, mask, raw in val_loader:
            # print(image.shape)
            # print(mask.shape)
            # print(raw.shape)
            image = image.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.float32)

            if name == "BASNet":
                pred, _, _, _, _, _, _, _ = model(image)
            elif name == "RendDANet":
                output = model(image)
                pred = torch.sigmoid(output['fine'])
            elif name == "RendUNet":
                output = model(image)
                coarse = F.interpolate(torch.sigmoid(output['coarse']), mode='nearest', scale_factor=8)
                stage1 = F.interpolate(torch.sigmoid(output['stage1']), mode='nearest', scale_factor=8)
                stage2 = F.interpolate(torch.sigmoid(output['stage2']), mode='nearest', scale_factor=8)
                stage3 = F.interpolate(torch.sigmoid(output['stage3']), mode='nearest', scale_factor=4)
                stage4 = F.interpolate(torch.sigmoid(output['stage4']), mode='nearest', scale_factor=2)
                pred = torch.sigmoid(output['fine'])
                batch_dice = dice_coeff(mask, pred).item()
                coarse = (coarse > 0.5).int().cpu().numpy().astype(np.uint8).reshape(384, 384) * 255
                stage1 = (stage1 > 0.5).int().cpu().numpy().astype(np.uint8).reshape(384, 384) * 255
                stage2 = (stage2 > 0.5).int().cpu().numpy().astype(np.uint8).reshape(384, 384) * 255
                stage3 = (stage3 > 0.5).int().cpu().numpy().astype(np.uint8).reshape(384, 384) * 255
                stage4 = (stage4 > 0.5).int().cpu().numpy().astype(np.uint8).reshape(384, 384) * 255
                final = (pred > 0.5).int().cpu().numpy().astype(np.uint8).reshape(384, 384) * 255

                img = Image.fromarray(coarse).convert('L')
                img.save("%s/%s/Seq_nearest/%d_%.2f_stage0.png" % (out_dir, name, seq, batch_dice))
                img = Image.fromarray(stage1).convert('L')
                img.save("%s/%s/Seq_nearest/%d_%.2f_stage1.png" % (out_dir, name, seq, batch_dice))
                img = Image.fromarray(stage2).convert('L')
                img.save("%s/%s/Seq_nearest/%d_%.2f_stage2.png" % (out_dir, name, seq, batch_dice))
                img = Image.fromarray(stage3).convert('L')
                img.save("%s/%s/Seq_nearest/%d_%.2f_stage3.png" % (out_dir, name, seq, batch_dice))
                img = Image.fromarray(stage4).convert('L')
                img.save("%s/%s/Seq_nearest/%d_%.2f_stage4.png" % (out_dir, name, seq, batch_dice))
                img = Image.fromarray(final).convert('L')
                img.save("%s/%s/Seq_nearest/%d_%.2f_stage5.png" % (out_dir, name, seq, batch_dice))
            else:
                pred = torch.sigmoid(model(image))

            batch_dice = dice_coeff(mask, pred).item()
            print(batch_dice)
            # raw = ((raw.cpu().float().numpy()) * 255).astype(np.uint8).reshape(384, 384, 3)
            raw = ((raw.cpu().float().numpy().reshape(384, 384, 3)) *255).astype(np.uint8)
            # print(raw.shape)
            mask = mask.cpu().int().numpy().astype(np.uint8).reshape(384, 384) * 255
            pred = (pred > 0.5).int().cpu().numpy().astype(np.uint8).reshape(384, 384) * 255

            for i in range(raw.shape[2]):
                temp = raw[:, :, i]
                img = Image.fromarray(temp).convert('L')
                img.save("%s/%s/Raw/%d_%d_%.2f.png" % (out_dir, name, seq, i, batch_dice))

            img = Image.fromarray(mask).convert('L')
            img.save("%s/%s/GT/%d_%.2f.png" % (out_dir, name, seq, batch_dice))
            img = Image.fromarray(pred).convert('L')
            img.save("%s/%s/Pred/%d_%.2f.png" % (out_dir, name, seq, batch_dice))
            # print(raw.shape, raw.mean())
            # print(pred.shape, pred.max())
            # print(mask.shape, mask.max())
            seq += 1
            # return

if __name__ == '__main__':
    output = inference()
    # exit(0)
