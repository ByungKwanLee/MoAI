from mmengine.structures import BaseDataElement as DC
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import ToTensor, ImageToTensor


@TRANSFORMS.register_module()
class SceneGraphFormatBundle(ImageToTensor):
    def __call__(self, results):
        results = super().__call__(results)

        if 'rel_fields' in results and len(results['rel_fields']) > 0:
            for key in results['rel_fields']:
                results[key] = DC(ToTensor(results[key]))
        if 'gt_scenes' in results:
            results['gt_scenes'] = DC(ToTensor(results['gt_scenes']))

        return results


@TRANSFORMS.register_module()
class PanopticSceneGraphFormatBundle(SceneGraphFormatBundle):
    def __call__(self, results):
        results = super().__call__(results)

        for key in ['all_gt_bboxes', 'all_gt_labels']:
            if key not in results:
                continue
            results[key] = DC(ToTensor(results[key]))

        return results
