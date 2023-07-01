import os
import torch
import numpy as np
from utils.pil2tensor import pil2tensor
from utils.tensor2pil import tensor2pil
from utils.device import device
from utils.ensure_package import ensure_package
import folder_paths
from utils.prompt2mask import prompt2mask

cached_groundingDINO_model = None
groundingdino_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./GroundingDINO_SwinT_OGC.py")
CATEGORY_NAME = "Shadel Nodes"
FOLDER_NAME = "grounding-dino"
model_path = folder_paths.models_dir

folder_paths.folder_names_and_paths[FOLDER_NAME] = ([os.path.join(model_path, FOLDER_NAME)], folder_paths.supported_pt_extensions)

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


