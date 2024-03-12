from torch import nn
from torch.nn import functional as F

from .build import register_model
from ..utils import configurable
from moai.utils.utils import *

# MoAI
from moai.load_moai import prepare_moai

class MoAI(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        cfg,
        moai_model,
        moai_processor,
        seg_model,
        seg_processor,
        od_model,
        od_processor,
        sgg_model,
        ocr_model,
    ):
        super().__init__()
        self.cfg = cfg
        self.moai_model = moai_model
        self.moai_processor = moai_processor
        self.seg_model = seg_model
        self.seg_processor = seg_processor
        self.od_model = od_model
        self.od_processor = od_processor
        self.sgg_model = sgg_model
        self.ocr_model = ocr_model

    @classmethod
    def from_config(cls, cfg):

        # MoAI
        if cfg['LLM']['LOAD_LLM']:
            moai_model, moai_processor, seg_model, seg_processor, od_model, od_processor, sgg_model, ocr_model \
                = prepare_moai(moai_path=cfg['RESUME_FROM'],
                bits=cfg['LLM']['BITS'],
                grad_ckpt=cfg['LLM']['GRAD_CKPT'],
                lora=cfg['LLM']['LORA'],
                dtype=cfg['LLM']['DTYPE'])
        else:
            moai_model, moai_processor = None, None

        return {
            "cfg": cfg,
            "moai_model": moai_model,
            "moai_processor": moai_processor,
            "seg_model": seg_model,
            "seg_processor": seg_processor,
            "od_model": od_model,
            "od_processor": od_processor,
            "sgg_model": sgg_model,
            "ocr_model": ocr_model,
            }

@register_model
def get_moai_model(cfg, **kwargs):
    return MoAI(cfg)