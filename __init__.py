from .ShadelNodes import NODE_CLASS_MAPPINGS
import os

cli_mode_flag = os.path.join(os.path.dirname(__file__), '.enable-cli-only-mode')

if not os.path.exists(cli_mode_flag):
    from .libs import ComfyuiInstallerImage
    WEB_DIRECTORY = "js"
    print(f"\n[ComfyuiInstallerImage] !!\n")
else:
    print(f"\n[ComfyUI-Manager] !! cli-only-mode is enabled !!\n")

__all__ = ['NODE_CLASS_MAPPINGS']
