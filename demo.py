# Loading Image
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor
image_path = "figures/moai_meme.jpg"
image = Resize(size=(490, 490), antialias=False)(pil_to_tensor(Image.open(image_path)))

# Instruction Prompt
prompt = "Describe this image."

# Loading MoAI
from moai.load_moai import prepare_moai
moai_model, moai_processor, seg_model, seg_processor, od_model, od_processor, sgg_model, ocr_model \
    = prepare_moai(moai_path='Bk-LEE/MoAI-7B', bits=4, grad_ckpt=False, lora=False, dtype='fp16')

# Pre-processing for MoAI
moai_inputs = moai_model.demo_process(image=image, 
                                    prompt=prompt, 
                                    processor=moai_processor,
                                    seg_model=seg_model,
                                    seg_processor=seg_processor,
                                    od_model=od_model,
                                    od_processor=od_processor,
                                    sgg_model=sgg_model,
                                    ocr_model=ocr_model,
                                    device='cuda:0')

# Generate
import torch
with torch.inference_mode():
    generate_ids = moai_model.generate(**moai_inputs, do_sample=True, top_p=0.95, max_new_tokens=128, use_cache=True)

# Decoding
decoded_text = moai_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]