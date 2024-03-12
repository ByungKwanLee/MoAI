import os
import logging
import math

from PIL import Image
from io import BytesIO
import base64

from detectron2.data import DatasetCatalog, MetadataCatalog
import pandas as pd

_PREDEFINED_SPLITS_PRETRAIN = {
    "mmbench": ["mmbench_dev_20230712.tsv"],
    "mmbench_cn": ["mmbench_dev_cn_20231003.tsv"],
    "mmbench_test": ["MMBench_TEST_EN_legacy.tsv"],
    # "mmbench_test": ["mmbench_test_20230712.tsv"],
    "mmbench_test_cn": ["MMBench_TEST_CN_legacy.tsv"]
}

def get_metadata(name):
    if name in ['mmbench', 'mmbench_test']:
        return {'gt_json': os.path.join(_coco_root, 'MMBench/mmbench_dev_20230712.tsv')}
    elif name in ['mmbench_cn', 'mmbench_test_cn']:
        return {'gt_json': os.path.join(_coco_root, 'MMBench/mmbench_dev_cn_20231003.tsv')}


evaluator_mapper = {'mmbench': 'mmbench', 'mmbench_cn': 'mmbench', 'mmbench_test': 'mmbench', 'mmbench_test_cn': 'mmbench'}

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
        questions = pd.read_table(os.path.join(root, arrow_path))
        arrs.append(questions)
    return arrs

def load_pretrain_data(arrow_root, meta, name, pretrain_arrows):
    ret = []
    questions = pretrain_arrows[0]

    all_options = ['A', 'B', 'C', 'D']

    def is_none(value):
        if value is None:
            return True
        if type(value) is float and math.isnan(value):
            return True
        if type(value) is str and value.lower() == 'nan':
            return True
        if type(value) is str and value.lower() == 'none':
            return True
        return False

    def get_options(row, options):
        parsed_options = []
        for option in options:
            option_value = row[option]
            if is_none(option_value):
                break
            parsed_options.append(option_value)
        return parsed_options

    for index, row in questions.iterrows():
        options = get_options(row, all_options)

        idx = row['index']
        question = row['question']
        hint = row['hint']
        image = Image.open(BytesIO(base64.b64decode(row['image'])))

        if not is_none(hint):
            question = hint + '\n' + question
        for option_char, option in zip(all_options[:len(options)], options):
            question = question + '\n' + option_char + '. ' + option

        ret.append( {
                    "question": question,
                    "question_id": idx,
                    "image": image,
                    "round_id": 0
                })

    assert len(ret), f"No images found in pretraining"
    return ret


def register_pretrain(
    name, metadata, arrow_root, arrow_paths
):
    semantic_name = name
    arrow_root = os.path.join(arrow_root, 'MMBench')
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
        # logger.warning("WARNING: Cannot find MMBench Dataset. Make sure datasets are accessible if you want to use them for training or evaluation.")        

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