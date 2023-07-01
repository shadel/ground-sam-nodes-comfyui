from .nodes.GroundingDINOLoader import GroundingDINOLoader
from .nodes.GroundedSAMSegNode import GroundedSAMSegNode

NODE_CLASS_MAPPINGS = {
    "Mask By Grounded SAM Text": GroundedSAMSegNode,
    "GroundingDINOLoader": GroundingDINOLoader,
}




