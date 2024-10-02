from ..config import CATEGORY_NAME
from ..utils.device import device
from ..utils.pil2tensor import pil2tensor
from ..utils.prompt2mask import prompt2mask
from ..utils.tensor2pil import np2pil, img_np_to_tensor, img_tensor_to_np


import numpy as np


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
        
        tensor_image_list = img_tensor_to_np(image)
        out_list = []

        for tensor_image in tensor_image_list:
            input_image_pil = np2pil(tensor_image)
            mask_image = prompt2mask(ground_model, sam_predictor, input_image_pil, prompt, box_threshold, text_threshold, num_boxes)
            # mask_image = pil2tensor(np.array(mask_image, dtype=np.uint8))

            out_list.append(mask_image)
        

        return (img_np_to_tensor(out_list), )