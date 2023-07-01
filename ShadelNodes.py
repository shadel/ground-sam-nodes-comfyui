import os
from nodes.GroundingDINOLoader import GroundingDINOLoader
from nodes.GroundedSAMSegNode import GroundedSAMSegNode
import folder_paths

cached_groundingDINO_model = None
groundingdino_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./GroundingDINO_SwinT_OGC.py")
CATEGORY_NAME = "Shadel Nodes"
FOLDER_NAME = "grounding-dino"
model_path = folder_paths.models_dir

folder_paths.folder_names_and_paths[FOLDER_NAME] = ([os.path.join(model_path, FOLDER_NAME)], folder_paths.supported_pt_extensions)

NODE_CLASS_MAPPINGS = {
    "Mask By Grounded SAM Text": GroundedSAMSegNode,
    "GroundingDINOLoader": GroundingDINOLoader,
}




