from .device import device


import numpy as np
import torch


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