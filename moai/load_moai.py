import torch

from .arch_moai import MoAIModel
from .arch.tokenization_internlm2 import MoAITokenizer
from peft import prepare_model_for_kbit_training

def prepare_moai(moai_path, bits, grad_ckpt, lora, dtype):

    # MoAI
    bnb_model_from_pretrained_args = {}
    if bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            torch_dtype=torch.bfloat16 if dtype=='bf16' else torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_skip_modules=["vision_tower", "vision_proj", "Plora_main", "moai", "output"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16 if dtype=='bf16' else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        ))

    # MoAIModel Loading
    moai_model = MoAIModel.from_pretrained(moai_path, **bnb_model_from_pretrained_args)
    moai_model.model.config.use_cache = False
    
    # parameter requires_grad 
    for param in moai_model.parameters():
        param.requires_grad=False

    # Gradient CKPT
    if bits in [4, 8] and not lora:
        # For only train
        if grad_ckpt:
            moai_model = prepare_model_for_kbit_training(moai_model,
                                                            use_gradient_checkpointing=grad_ckpt,
                                                            gradient_checkpointing_kwargs={"use_reentrant": True})
    elif bits in [4, 8] and lora:
        raise Exception("MoAI does not have any plan in lora with bit quantization")
    elif not bits in [4, 8] and lora:
        raise Exception("MoAI does not have any plan in lora without bit quantization")
    elif not bits in [4, 8] and not lora:
        raise Exception("MoAI does not have any plan in full training without lora and bit quantization")
    else:
        raise Exception("No Way!")

    # bfloat16/float16 conversion 
    for param in moai_model.parameters():
        if 'float32' in str(param.dtype).lower() or 'float16' in str(param.dtype).lower():
            param.data = param.data.to(torch.bfloat16 if dtype=='bf16' else torch.float16)

    # Training MoAI
    for name, param in moai_model.named_parameters():
        if 'moai' in name:
            param.requires_grad_(True)
    
    # Post-Processing for <image> Token
    moai_processor = MoAITokenizer.from_pretrained(moai_path, padding_side='left')
    moai_processor.add_tokens("<image>", special_tokens=True)
    moai_model.resize_token_embeddings(len(moai_processor))
    moai_model.config.image_token_index = moai_processor("<image>", add_special_tokens=False, return_tensors='pt').input_ids.item()
    moai_model.config.ignore_index = -100
    moai_model.config.pad_token_id = -100
    
    # load Mask2Former fine-tuned on COCO panoptic segmentation
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-base-coco-panoptic').cuda()
    seg_processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-base-coco-panoptic')
    
    # load OWLV2 for open world object detection    
    from transformers import AutoProcessor, Owlv2ForObjectDetection
    od_model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble').cuda()
    od_processor = AutoProcessor.from_pretrained('google/owlv2-base-patch16-ensemble')

    # load SGG
    import sys
    sys.path.append('moai/sgg')
    from mmengine import Config
    from mmdet.apis import init_detector
    cfg = Config.fromfile('moai/sgg/configs/psgtr/psgtr_r50_psg_inference.py')
    sgg_model = init_detector(cfg, 'moai/sgg/checkpoints/psgtr_r50_epoch_60.pth', palette="none").cuda()

    # load OCR
    from paddleocr import PaddleOCR
    ocr_model = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)
    
    return moai_model, moai_processor, seg_model, seg_processor, od_model, od_processor, sgg_model, ocr_model

# for name, param in moai_model.named_parameters(): print(f"{name}: {param.dtype} {param.requires_grad}")