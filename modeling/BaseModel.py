import logging

import torch.nn as nn
logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        return outputs
    
    # for name, param in self.model.moai_model.named_parameters(): print(f"{name}: {param.dtype} {param.requires_grad}")