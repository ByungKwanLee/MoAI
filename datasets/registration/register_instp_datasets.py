# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu), Ziyi Dou (zdou@cs.ucla.edu)
# --------------------------------------------------------
import os
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
import pyarrow as pa

_PREDEFINED_SPLITS_PRETRAIN = {
    # filt coco2017 val
    "instp_train": ["instruction_pretrain_558k.arrow"],
    "instp_val": ["coco_caption_karpathy_test.arrow"],
    "instp_captioning_val": ["coco_caption_karpathy_test.arrow"],
    "instp_val2017": ["coco_caption_karpathy_val2017.arrow"],
    "instp_captioning_val2017": ["coco_caption_karpathy_val2017.arrow"],
    # the following is for local testing
}

def get_metadata(name):
    if name in ['instp_captioning_val', 'instp_captioning_val2017']:
        return {'gt_json': os.path.join(_coco_root, 'coco/annotations/captions_val2014.json')}
    else:
        return {}

evaluator_mapper = {'instp_val': 'retrieval', 'instp_train': 'retrieval', 'instp_captioning_val': 'captioning', 'instp_val2017': 'retrieval', 'instp_captioning_val2017': 'captioning'}
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
        arr = pa.ipc.RecordBatchFileReader(
                        pa.memory_map(os.path.join(root, arrow_path), "r")
                    ).read_all()

        arrs.append(arr)
    return arrs

def load_pretrain_data(arrow_root, meta, name, pretrain_arrows):
    ret = []

    image_id = 0
    arr_id = 0
    for arr in pretrain_arrows:
        arr_len = len(arr)
        cur_id = 0
        for i in range(arr_len):
            captions = arr['caption'][i].as_py()
            image_id = arr['image_id'][i].as_py()
            if not isinstance(image_id, int):
                image_id = int(image_id.split('/')[-1].split('.')[0])

            if 'val' in name:
                ret.append( {
                    "image_id": image_id,
                    "captions": captions,
                    "arr_id": arr_id,
                    "cur_id": cur_id,
                })
            else:
                ret.append( {
                    "image_id": image_id,
                    "captions": [captions],
                    "arr_id": arr_id,
                    "cur_id": cur_id,
                })
            cur_id += 1
            image_id += 1

        arr_id += 1

    assert len(ret), f"No images found in pretraining"
    return ret


def register_pretrain(
    name, metadata, arrow_root, arrow_paths
):
    # the name is "coco_2017_train/val_caption_only"
    semantic_name = name
    arrow_root = os.path.join(arrow_root, 'llava_dataset')
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
        # logger.warning("WARNING: Cannot find InsturctionDataset. Make sure datasets are accessible if you want to use them for training or evaluation.")        

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