from ..config import CATEGORY_NAME
from ..utils.device import device
from ..utils.ensure_package import ensure_package


import torch


import os
import folder_paths

class WebUIAPIConfig:
  def __init__(self, name, age):
    self.name = name
    self.age = age

class WebUIAPINode:
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