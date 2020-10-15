import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_transform import HorizontalFlip, VerticalFlip, Rotate, ToTensor, CenterCrop, Normalize


class NPCDataset(Dataset):

    def __init__(self, img_dir, mask_dir, mode='test'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.images = list(sorted(os.listdir(img_dir)))
        self.masks = list(sorted(os.listdir(mask_dir)))
        self.cc = CenterCrop(size=224)
        self.hf = HorizontalFlip(p=0.5)
        self.vf = VerticalFlip(p=0.5)
        self.rt = Rotate(degrees=(90, 180, 270))
        self.tt = ToTensor()
        self.nl = Normalize()

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = np.load(os.path.join(self.img_dir, self.images[item]))
        mask = np.load(os.path.join(self.mask_dir, self.masks[item]))

        # here image & mask are ndarray of 384*384*3 with dtype float64

        if self.mode == "train":

            # image, mask = self.cc(image, mask)
            image, mask = self.hf(image, mask)
            image, mask = self.vf(image, mask)
            image, mask = self.rt(image, mask)
            image, mask = self.tt(image, mask)
            image, mask = self.nl(image, mask)

        elif self.mode == 'val':

            image, mask = self.tt(image, mask)
            image, mask = self.nl(image, mask)

        elif self.mode == 'test':

            image, mask = self.tt(image, mask)
            temp = torch.from_numpy(image.numpy().transpose((1, 2, 0)))
            image, mask = self.nl(image, mask)

            return image, mask, temp

        else:
            print("invalid transform mode")

        return image, mask


def get_dataloader(img_dir, mask_dir, batch_size, num_workers, mode="train"):

    if mode == "train":
        train_dataset = NPCDataset(img_dir, mask_dir, mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return train_dataloader
    elif mode == "test":
        val_dataset = NPCDataset(img_dir, mask_dir, mode='test')
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        return val_dataloader
    else:
        val_dataset = NPCDataset(img_dir, mask_dir, mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        return val_dataloader


# if __name__ == "__main__":
#
#     train_image_dir = "../data/Multi_V1/train/image"
#     train_label_dir = "../data/Multi_V1/train/label"
#     val_image_dir = "../data/Multi_V1/val/image"
#     val_label_dir = "../data/Multi_V1/val/label"
#
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     train_loader = get_dataloader(train_image_dir, train_label_dir, batch_size=1, num_workers=1, mode="train")
#     val_loader = get_dataloader(val_image_dir, val_label_dir, batch_size=1, num_workers=1, mode="val")
#
#     train_dataset = NPCDataset(train_image_dir, train_label_dir, mode="train")
#     image, mask = train_dataset[0]
#     print(image.shape)
#     print(mask.shape)
