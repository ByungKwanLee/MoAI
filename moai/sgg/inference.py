from __future__ import annotations

import cv2

from PIL import Image

import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmengine import Config
from utils import show_result, make_gif

def update_input_image(image):
    image = np.array(image)
    scale = 800 / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image

class Model:
    def __init__(self):
        model_ckt ='checkpoints/psgtr_r50_epoch_60.pth'
        cfg = Config.fromfile('configs/psgtr/psgtr_r50_psg_inference.py')
        self.model = init_detector(cfg, model_ckt, palette="none").cuda()

    def infer(self, input_image, num_rel):
        result = inference_detector(self.model, input_image, None)
        for res in result:
            displays = show_result(input_image,
                                res,
                                is_one_stage=True,
                                num_rel=num_rel,
                                show=True,
                                out_file="cooking.jpg"
                                )
            print(displays)
        # gif = make_gif(displays[:10] if len(displays) > 10 else displays)
        return result


def main():
    num_rel = 10
    model = Model()
    input_image = Image.open("images/images_neymar-jr-angers-x-psg-160121.jpg")
    resized_image = update_input_image(input_image)

    result = model.infer([resized_image, resized_image], num_rel)

if __name__ == '__main__':
    main()