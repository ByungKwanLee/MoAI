"""
See: https://mmdetection.readthedocs.io/en/v2.19.0/tutorials/customize_models.html
"""
from mmdet.models import StandardRoIHead
from mmdet.registry import MODELS


@MODELS.register_module()
class SceneGraphRoIHead(StandardRoIHead):
    def __init__(self, param, **kwargs):
        super().__init__(**kwargs)
        self.param = self.param
