from PIL import Image, ImageFilter, ImageOps
import os
import torch
import numpy as np
import math
from torchvision import transforms
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import torch.nn.functional as torchfn
import subprocess
import sys

DELIMITER = '|'
cached_clipseg_model = None
VERY_BIG_SIZE = 1024 * 1024

package_list = None
def update_package_list():
    import sys
    import subprocess

    global package_list
    package_list = [r.decode().split('==')[0] for r in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).split()]

def ensure_package(package_name, import_path):
    global package_list
    if package_list == None:
        update_package_list()

    if package_name not in package_list:
        print("(First Run) Installing missing package %s" % package_name)
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', import_path])
        update_package_list()

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t

def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t

def tensor2batch(t: torch.Tensor, bs: torch.Size) -> torch.Tensor:
    if len(t.size()) < len(bs):
        t = t.unsqueeze(3)
    if t.size()[0] < bs[0]:
        t.repeat(bs[0], 1, 1, 1)
    dim = bs[3]
    if dim == 1:
        return tensor2mask(t)
    elif dim == 3:
    elif dim == 4:
        return tensor2rgba(t)

def tensors2common(t1: torch.Tensor, t2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    t1s = t1.size()
    t2s = t2.size()
    if len(t1s) < len(t2s):
        t1 = t1.unsqueeze(3)
    elif len(t1s) > len(t2s):
        t2 = t2.unsqueeze(3)

    if len(t1.size()) == 3:
        if t1s[0] < t2s[0]:
            t1 = t1.repeat(t2s[0], 1, 1)
        elif t1s[0] > t2s[0]:
            t2 = t2.repeat(t1s[0], 1, 1)
    else:
        if t1s[0] < t2s[0]:
            t1 = t1.repeat(t2s[0], 1, 1, 1)
        elif t1s[0] > t2s[0]:
            t2 = t2.repeat(t1s[0], 1, 1, 1)

    t1s = t1.size()
    t2s = t2.size()
    if len(t1s) > 3 and t1s[3] < t2s[3]:
        return tensor2batch(t1, t2s), t2
    elif len(t1s) > 3 and t1s[3] > t2s[3]:
        return t1, tensor2batch(t2, t1s)
    else:
        return t1, t2

class GroundedSAMSegNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "precision": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "normalize": (["no", "yes"],),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    FUNCTION = "get_mask"

    CATEGORY = "Shadel Nodes"

    def get_mask(self, image, prompt, negative_prompt, precision, normalize):

        model = self.load_model()
        image = tensor2rgb(image)
        B, H, W, _ = image.shape
        # clipseg only works on square images, so we'll just use the larger dimension
        # TODO - Should we pad instead of resize?
        used_dim = max(W, H)

        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((used_dim, used_dim), antialias=True) ])
        img = transform(image.permute(0, 3, 1, 2))

        prompts = prompt.split(DELIMITER)
        negative_prompts = negative_prompt.split(DELIMITER) if negative_prompt != '' else []
        with torch.no_grad():
            # Optimize me: Could do positive and negative prompts as part of one batch
            dup_prompts = [item for item in prompts for _ in range(B)]
            preds = model(img.repeat(len(prompts), 1, 1, 1), dup_prompts)[0]
            dup_neg_prompts = [item for item in negative_prompts for _ in range(B)]
            negative_preds = model(img.repeat(len(negative_prompts), 1, 1, 1), dup_neg_prompts)[0] if len(negative_prompts) > 0 else None

        preds = torch.nn.functional.interpolate(preds, size=(H, W), mode='nearest')
        preds = torch.sigmoid(preds)
        preds = preds.reshape(len(prompts), B, H, W)
        mask = torch.max(preds, dim=0).values

        if len(negative_prompts) > 0:
            negative_preds = torch.nn.functional.interpolate(negative_preds, size=(H, W), mode='nearest')
            negative_preds = torch.sigmoid(negative_preds)
            negative_preds = negative_preds.reshape(len(negative_prompts), B, H, W)
            mask_neg = torch.max(negative_preds, dim=0).values
            mask = torch.min(mask, 1. - mask_neg)

        if normalize == "yes":
            mask_min = torch.min(mask)
            mask_max = torch.max(mask)
            mask_range = mask_max - mask_min
            mask = (mask - mask_min) / mask_range
        thresholded = torch.where(mask >= precision, 1., 0.)
        # import code
        # code.interact(local=locals())
        return (thresholded.to(device=image.device), mask.to(device=image.device),)

    def load_model(self):
        global cached_clipseg_model
        if cached_clipseg_model == None:
            ensure_package("clipseg", "clipseg@git+https://github.com/timojl/clipseg.git@bbc86cfbb7e6a47fb6dae47ba01d3e1c2d6158b0")
            from clipseg.clipseg import CLIPDensePredT
            model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            model.eval()

            d64_file = self.download_and_cache('rd64-uni-refined.pth', 'https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd64-uni-refined.pth')
            d16_file = self.download_and_cache('rd16-uni.pth', 'https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd16-uni.pth')
            # Use CUDA if it's available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(d64_file, map_location=device), strict=False)
            model = model.eval().to(device=device)
            cached_clipseg_model = model
        return cached_clipseg_model

    def download_and_cache(self, cache_name, url):
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_cache')
        os.makedirs(cache_dir, exist_ok=True)

        file_name = os.path.join(cache_dir, cache_name)
        if not os.path.exists(file_name):
            print(f'Downloading and caching file: {cache_name}')
            with open(file_name, 'wb') as file:
                import requests
                r = requests.get(url, stream=True)
                r.raise_for_status()
                for block in r.iter_content(4096):
                    file.write(block)
            print('Finished downloading.')

        return file_name

NODE_CLASS_MAPPINGS = {
    "Mask By Text": GroundedSAMSegNode,
}
