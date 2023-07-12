import torchvision.transforms as transforms
import numpy as np
from typing import Tuple
import random
import torch


class SageTransform(object):
    def __init__(self, aug=False, rgb_to_greyscale=False):
        self.aug = aug
        if rgb_to_greyscale:
            fun_rgb_trans = transforms.Grayscale(num_output_channels=1)
        else:
            # these parameters are from augmentation in vicreg
            fun_rgb_trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        self.rgb_transform = lambda rh, rw: transforms.Compose(
            [
                transforms.ToTensor(),
                fun_rgb_trans,
                # crop the image to match the size of thermal image\
                # keep the same aspect ratio 4/3
                # TODO try to train the NN with different ratio
                transforms.CenterCrop((rh, rw)),
                # transforms.CenterCrop((1800, 2400)),
                transforms.Resize((600, 800))
            ]
        )
        self.thermal_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #  std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image_pair: Tuple[np.ndarray, np.ndarray, str]):
        ratio = 4 / 3
        rh = int(image_pair[0].shape[0] / 1.1)
        rw = int(rh * ratio)
        rgb_out = self.rgb_transform(rh, rw)(image_pair[0])
        thermal_out = self.thermal_transform(image_pair[1])
        if self.aug:
            rgb_size = np.array([600, 800])
            ir_size = np.array([252, 336])
            ratio = [rgb_size[i] / ir_size[i] for i in range(2)]
            new_ir_dim = np.array(transforms.RandomCrop(200).get_params(thermal_out, (200, 200)))
            new_rgb_dim = (np.array(new_ir_dim) * ratio[0]).astype(int)
            rgb_out = transforms.functional.crop(rgb_out, *new_rgb_dim)
            thermal_out = transforms.functional.crop(thermal_out, *new_ir_dim)
            rand_t = RandomChoice([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                        ])
            rgb_out, thermal_out = rand_t([rgb_out, thermal_out])
        return rgb_out, thermal_out, image_pair[2]


class SageRGBTransform(object):
    def __init__(self):
        self.rgb_orig_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # these parameters are from augmentation in vicreg
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                # crop the image to match the size of thermal image\
                # keep the same aspect ratio 4/3
                # TODO try to train the NN with different ratio
                transforms.CenterCrop((1800, 2400)),
                transforms.Resize((600, 800))
            ]
        )
        # this shall transform the rgb image
        self.rgb_aug_transform = transforms.Compose(
            [
                transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225]),
                transforms.Grayscale(num_output_channels=1),
                transforms.CenterCrop((1440, 1920)), # 0.8x (1800, 2400)
                transforms.Resize((252, 336))
                
            ]
        )

    def __call__(self, image_pair: Tuple[np.ndarray, str]):
        rgb_out = self.rgb_orig_transform(image_pair[0])
        rgb_aug_out = self.rgb_aug_transform(image_pair[0])
        return rgb_out, rgb_aug_out, image_pair[1]
    
    
class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        return [t(img) for img in imgs]
