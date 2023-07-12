import torchvision.transforms as transforms
import numpy as np
from typing import Tuple
import random
import torch


class SageTransform(object):
    def __init__(self, aug=False, rgb_to_greyscale=False, compare_rgb_gs=False):
        self.aug = aug
        self.rgb_to_greyscale = rgb_to_greyscale
        self.compare_rgb_gs = compare_rgb_gs
        self.rgb_outsize = (600, 800)
        self.ir_outsize = (252, 336)
        # assign functional
        self.left_transform = self.rgb_transform
        if compare_rgb_gs:
            self.right_transform = self.rgb_transform
        else:
            self.right_transform = self.thermal_transform
        
        
    def __call__(self, image_pair: Tuple[np.ndarray, np.ndarray, str]):
        ratio = 4 / 3
        rh = int(image_pair[0].shape[0] / 1.1)
        rw = int(rh * ratio)
        left_trans_func = self.left_transform(rh, rw,
                                              *self.rgb_outsize,
                                              self.rgb_to_greyscale)
        rgb_out = left_trans_func(image_pair[0])
        if self.compare_rgb_gs:
            right_trans_func = self.right_transform(int(rh*0.8), int(rw*0.8),
                                                    *self.ir_outsize,
                                                    to_grayscale=True)
        else:
            right_trans_func = self.right_transform()
        thermal_out = right_trans_func(image_pair[1])
        if self.aug:
            rgb_size = self.rgb_outsize
            ir_size = self.ir_outsize
            ratio = [rgb_size[i] / ir_size[i] for i in range(2)]
            new_ir_dim = np.array(transforms.RandomCrop(200).get_params(thermal_out,
                                                                        (200, 200)))
            new_rgb_dim = (np.array(new_ir_dim) * ratio[0]).astype(int)
            rgb_out = transforms.functional.crop(rgb_out, *new_rgb_dim)
            thermal_out = transforms.functional.crop(thermal_out, *new_ir_dim)
            rand_t = RandomChoice(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ]
            )
            rgb_out, thermal_out = rand_t([rgb_out, thermal_out])
        return rgb_out, thermal_out, image_pair[2]
    
    
    def rgb_transform(self, crop_h, crop_w, out_h, out_w, to_grayscale):
        if to_grayscale:
            fun_rgb_trans = transforms.Grayscale(num_output_channels=1)
        else:
            # these parameters are from augmentation in vicreg
            fun_rgb_trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        trans_func = transforms.Compose(
            [
                transforms.ToTensor(),
                fun_rgb_trans,
                # crop the image to match the size of thermal image\
                # keep the same aspect ratio 4/3
                # TODO try to train the NN with different ratio
                transforms.CenterCrop((crop_h, crop_w)),
                # transforms.CenterCrop((1800, 2400)),
                transforms.Resize((out_h, out_w))
            ]
        )
        return trans_func

    
    def thermal_transform(self):
        trans_func = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        return trans_func


#     def grayscale_transform(rh, rw, out_size):
#         trans_func = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Grayscale(num_output_channels=1),
#                 transforms.CenterCrop((rh, rw)), # 0.8x (1800, 2400)
#                 transforms.Resize(out_size)
#             ]
#         )
#         return trans_func
        

class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        return [t(img) for img in imgs]
