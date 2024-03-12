import os
import json
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog

_PREDEFINED_SPLITS_PRETRAIN = {
    "mm-vet": ["llava-mm-vet.json"],
}

def get_metadata(name):
    if name in ['mm-vet']:
        return {'gt_json': os.path.join(_coco_root, 'mm-vet/llava-mm-vet.json')}
    else:
        return {}

evaluator_mapper = {'mm-vet': 'mm-vet'}
def load_pretrain_arrows(root, arrow_paths):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    arrs = []
    for arrow_path in arrow_paths:
        questions = json.load(open(os.path.join(root, arrow_path), "r"))
        arrs.append(questions)
    return arrs

def load_pretrain_data(arrow_root, meta, name, pretrain_arrows):
    ret = []
    questions = pretrain_arrows[0]

    for question in questions:

        image_id = question['image']
        text = question['text']
        question_id = question['question_id']

        ret.append( {
                    "image_id": image_id,
                    "question": text,
                    "question_id": question_id,
                })

    assert len(ret), f"No images found in pretraining"
    return ret


def register_pretrain(
    name, metadata, arrow_root, arrow_paths
):
    semantic_name = name
    arrow_root = os.path.join(arrow_root, 'mm-vet')
    if os.path.exists(arrow_root):
        pretrain_arrows = load_pretrain_arrows(arrow_root, arrow_paths)
        DatasetCatalog.register(
            semantic_name,
            lambda: load_pretrain_data(arrow_root, metadata, name, pretrain_arrows),
        )
        MetadataCatalog.get(semantic_name).set(
            arrow_root=arrow_root,
            evaluator_type=evaluator_mapper[name],
            arrows=pretrain_arrows,
            **metadata,
        )
    else:
        logger = logging.getLogger(__name__)
        # logger.warning("WARNING: Cannot find mm-vet Dataset. Make sure datasets are accessible if you want to use them for training or evaluation.")        

def register_all_pretrain(root):
    for (
        prefix,
        arrow_paths,
    ) in _PREDEFINED_SPLITS_PRETRAIN.items():
        register_pretrain(
            prefix,
            get_metadata(prefix),
            root,
            arrow_paths,
        )


# _root = os.getenv("VLDATASET", "datasets") #may need a different root name?
_root = os.getenv("DATASET2", "datasets") #may need a different root name?
_coco_root = os.getenv("DATASET", "datasets") #may need a different root name?
register_all_pretrain(_root)