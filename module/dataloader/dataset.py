import os
import module.dataloader.ext_transforms as et
import torch.utils.data as data
from PIL import Image
import numpy as np
from .constant import train_id_to_color, id_to_train_id, syn_id_to_train_id
# for synthia
import imageio
imageio.plugins.freeimage.download()


class CityscapesGTA5(data.Dataset):
    """GTA5 Synthetic Dataset."""

    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 1.5)),
        et.ExtResize((1024, 2048)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtResize((1024, 2048)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, root, datalist, split='train', transform=None):
        self.root = os.path.expanduser(root)
        if split not in ['train', 'test', 'val', 'active-label', 'active-ulabel', 'custom-set']:
            raise ValueError('Invalid split for mode!')
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.train_transform if split in ["train", "active-label"] else self.val_transform

        self.split = split
        self.im_idx = []
        if datalist is not None:
            valid_list = np.loadtxt(datalist, dtype='str')
            for img_fname, lbl_fname, *rest in valid_list:
                img_fullname = os.path.join(self.root, img_fname)
                lbl_fullname = os.path.join(self.root, lbl_fname)
                self.im_idx.append([img_fullname, lbl_fullname])

    @classmethod
    def encode_target(cls, target):
        return id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return train_id_to_color[target]

    def __getitem__(self, index):
        img_fname, lbl_fname = self.im_idx[index]
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        image, target = self.transform(image, [target])
        target = self.encode_target(target[0])
        sample = {'images': image, 'labels': target, 'fnames': self.im_idx[index], 'index': index }
        return sample

    def __len__(self):
        return len(self.im_idx)


class SYNTHIA(data.Dataset):
    """SYNTHIA Dataset."""

    train_transform = et.ExtCompose([
        et.ExtResize((1024, 2048)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtResize((1024, 2048)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, root, datalist='./dataloader/init_data/SYNTHIA/train.txt', split='train', transform=None):
        self.root = os.path.expanduser(root)
        if split not in ['train', 'test', 'val', 'active-label', 'active-ulabel', 'custom-set']:
            raise ValueError('Invalid split for mode!')
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.train_transform if split in ["train", "active-label"] else self.val_transform

        self.split = split
        self.im_idx = []
        if datalist is not None:
            valid_list = np.loadtxt(datalist, dtype='str')
            for img_fname, lbl_fname, *rest in valid_list:
                img_fullname = os.path.join(self.root, img_fname)
                lbl_fullname = os.path.join(self.root, lbl_fname)
                self.im_idx.append([img_fullname, lbl_fullname])

    @classmethod
    def encode_target(cls, target):
        target_copy = 255 * np.ones(np.array(target).shape, dtype=np.uint8)
        for k, v in enumerate(syn_id_to_train_id):
            target_copy[target == k] = v
        return target_copy

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return train_id_to_color[target]

    def __getitem__(self, index):
        img_fname, lbl_fname = self.im_idx[index]
        image = Image.open(img_fname).convert('RGB')
        target = np.asarray(imageio.imread(lbl_fname, format='PNG-FI'))[:, :, 0]
        target = np.array(target, dtype=np.uint8)
        target = Image.fromarray(target)
        image, target = self.transform(image, [target])
        target = self.encode_target(target[0])
        sample = {'images': image, 'labels': target, 'fnames': self.im_idx[index]}
        return sample

    def __len__(self):
        return len(self.im_idx)