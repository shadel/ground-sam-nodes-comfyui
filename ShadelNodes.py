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
import folder_paths

DELIMITER = '|'
cached_groundingDINO_model = None
VERY_BIG_SIZE = 1024 * 1024
device = "cuda" if torch.cuda.is_available() else "cpu"
groundingdino_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./GroundingDINO_SwinT_OGC.py")
CATEGORY_NAME = "Shadel Nodes"
FOLDER_NAME = "grounding-dino"
model_path = folder_paths.models_dir

folder_paths.folder_names_and_paths[FOLDER_NAME] = ([os.path.join(model_path, FOLDER_NAME)], folder_paths.supported_pt_extensions)

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

def prompt2mask(grounding_model, sam_predictor, original_image, caption, box_threshold=0.25, text_threshold=0.25, num_boxes=2):
    
    import groundingdino.datasets.transforms as T
    from groundingdino.util.inference import  predict
    from segment_anything.utils.amg import remove_small_regions
    
    def image_transform_grounding(init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(init_image, None)  # 3, h, w
        return init_image, image

    image_np = np.array(original_image, dtype=np.uint8)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    _, image_tensor = image_transform_grounding(original_image)
    boxes, logits, phrases = predict(grounding_model,
                                     image_tensor, caption, box_threshold, text_threshold, device='cpu')
    print(logits)
    print('number of boxes: ', boxes.size(0))
    # from PIL import Image, ImageDraw, ImageFont
    H, W = original_image.size[1], original_image.size[0]
    boxes = boxes * torch.Tensor([W, H, W, H])
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]

    final_m = torch.zeros((image_np.shape[0], image_np.shape[1]))

    if boxes.size(0) > 0:
        sam_predictor.set_image(image_np)

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2])
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )

        # remove small disconnected regions and holes
        fine_masks = []
        for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
            fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
        masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
        masks = torch.from_numpy(masks)

        num_obj = min(len(logits), num_boxes)
        for obj_ind in range(num_obj):
            # box = boxes[obj_ind]

            m = masks[obj_ind][0]
            final_m += m
    final_m = (final_m > 0).to('cpu').numpy()
    # print(final_m.max(), final_m.min())
    return np.dstack((final_m, final_m, final_m)) * 255
    


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class GroundedSAMSegNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ground_model": ("GROUNDING_DINO_MODEL",),
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "box_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_boxes": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_mask"

    CATEGORY = CATEGORY_NAME

    def get_mask(self, ground_model, sam_model, image, prompt, box_threshold, text_threshold, num_boxes):

        from segment_anything import SamPredictor
        sam_predictor = SamPredictor(sam_model.to(device))

        input_image_pil = tensor2pil(image)
        mask_image = prompt2mask(ground_model, sam_predictor, input_image_pil, prompt, box_threshold, text_threshold, num_boxes)

        mask_image = pil2tensor(np.array(mask_image, dtype=np.uint8))
        return (mask_image, )

class GroundingDINOLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (folder_paths.get_filename_list(FOLDER_NAME), )}}

    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )
    FUNCTION = "load_model"

    CATEGORY = CATEGORY_NAME

    def load_model(self, model_name):
        modelname = folder_paths.get_full_path(FOLDER_NAME, model_name)

        print(f"Loads grounding-dino model: {modelname}")
        global cached_groundingDINO_model
        if cached_groundingDINO_model == None:
            ensure_package("GroundingDINO", "git+https://github.com/IDEA-Research/GroundingDINO.git")

            # d64_file = self.download_and_cache('groundingdino_swint_ogc.pth', 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth')
            # Use CUDA if it's available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = self.load_groundingdino_model(groundingdino_config_file, modelname).to(device=device)
            cached_groundingDINO_model = model
        return (cached_groundingDINO_model, )

    def download_and_cache(self, cache_name, url):
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), FOLDER_NAME)
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
    
    def load_groundingdino_model(self, model_config_path, model_checkpoint_path):
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict

        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model


NODE_CLASS_MAPPINGS = {
    "Mask By Grounded SAM Text": GroundedSAMSegNode,
    "GroundingDINOLoader": GroundingDINOLoader,
}
