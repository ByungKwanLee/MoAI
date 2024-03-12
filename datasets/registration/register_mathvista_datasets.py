import os
import json
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
import pandas as pd

_PREDEFINED_SPLITS_PRETRAIN = {
    "mathvista_testmini": ["testmini-00000-of-00001-725687bf7a18d64b.parquet"],
    "mathvista_test": ["test-00000-of-00002-6b81bd7f7e2065e6.parquet", "test-00001-of-00002-6a611c71596db30f.parquet"]
}

def get_metadata(name):
    if name in ['mathvista_testmini', 'mathvista_test']:
        return {'gt_json': os.path.join(_coco_root, 'MathVista/annot_testmini.json')}

evaluator_mapper = {'mathvista_testmini': 'mathvista', 'mathvista_test': 'mathvista'}

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
        questions = pd.read_parquet(os.path.join(root, arrow_path), engine='pyarrow')
        arrs.append(questions)
    return arrs

def load_pretrain_data(arrow_root, meta, name, pretrain_arrows):
    ret = []
    for questions in pretrain_arrows:
        for i, question in questions.iterrows():
            image_id = question['image']
            question['image_id'] = image_id
            ret.append(question.to_dict())

    assert len(ret), f"No images found in pretraining"
    return ret

def register_pretrain(
    name, metadata, arrow_root, arrow_paths
):
    semantic_name = name
    arrow_root = os.path.join(arrow_root, 'MathVista')
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
        logger.warning("WARNING: Cannot find MathVista Dataset. Make sure datasets are accessible if you want to use them for training or evaluation.")        

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