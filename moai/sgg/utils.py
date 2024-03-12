from typing import Tuple
import PIL
import mmcv
import numpy as np
from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.evaluation import INSTANCE_OFFSET
from PIL import Image
import os

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged', 'background'
]

PREDICATES = [
    'over',
    'in front of',
    'beside',
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]


def get_colormap(num_colors: int):
    return (np.resize(colormap(), (num_colors, 3))).tolist()


def draw_text(
    viz_img: VisImage = None,
    text: str = None,
    x: float = None,
    y: float = None,
    color: Tuple[float, float, float] = [0, 0, 0],
    size: float = 10,
    padding: float = 5,
    box_color: str = 'black',
    font: str = None,
) -> float:
    text_obj = viz_img.ax.text(
        x,
        y,
        text,
        size=size,
        # family="sans-serif",
        bbox={
            'facecolor': box_color,
            'alpha': 0.8,
            'pad': padding,
            'edgecolor': 'none',
        },
        verticalalignment='top',
        horizontalalignment='left',
        color=color,
        zorder=10,
        rotation=0,
    )
    viz_img.get_image()
    text_dims = text_obj.get_bbox_patch().get_extents()

    return text_dims.width

def label_duplicates(lst):
    duplicates_count = {}
    
    labeled_list = []
    
    for item in lst:
        if lst.count(item) > 1:
            if item not in duplicates_count:
                duplicates_count[item] = 1
            else:
                duplicates_count[item] += 1
            labeled_list.append(f"{item} (#{duplicates_count[item]})")
        else:
            labeled_list.append(item)
    
    return labeled_list

def show_result(img,
                result,
                is_one_stage,
                num_rel=20,
                show=False,
                out_dir=None,
                out_file=None):
    # Load image
    # img = mmcv.imread(img)
    # img = img.copy()  # (H, W, 3)
    # img_h, img_w = img.shape[:-1]
    
    # Decrease contrast
    # img = PIL.Image.fromarray(img)
    # converter = PIL.ImageEnhance.Color(img)
    # img = converter.enhance(0.01)
    # if out_file is not None:
    #     mmcv.imwrite(np.asarray(img), 'bw'+out_file)

    # Draw masks
    pan_results = result.pan_results

    ids = np.unique(pan_results)[::-1]
    num_classes = 133
    legal_indices = (ids != num_classes)  # for VOID label
    ids = ids[legal_indices]

    # Get predicted labels
    labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    labels = [CLASSES[l] for l in labels] # what objects are in the image

    #For psgtr
    rel_obj_labels = result.labels
    rel_obj_labels = [CLASSES[l - 1].replace('-merged','').replace('-other','').replace('-stuff','') for l in rel_obj_labels] # what objects are in the image
    rel_obj_labels = label_duplicates(rel_obj_labels)
    # (N_m, H, W)
    # segms = pan_results[None] == ids[:, None, None]
    # # Resize predicted masks
    # segms = [
    #     mmcv.image.imresize(m.astype(float), (img_w, img_h)) for m in segms
    # ]
    # One stage segmentation
    # masks = result.masks # segmentation masks

    # Choose colors for each instance in coco
    # colormap_coco = get_colormap(len(masks)) if is_one_stage else get_colormap(len(segms))
    # colormap_coco = (np.array(colormap_coco) / 255).tolist()

    # Viualize masks
    # viz = Visualizer(img)
    # viz.overlay_instances(
    #     labels=rel_obj_labels if is_one_stage else labels,
    #     masks=masks if is_one_stage else segms,
    #     assigned_colors=colormap_coco,
    # )
    # viz_img = viz.get_output().get_image()
    # if out_file is not None:
    #     mmcv.imwrite(viz_img, out_file)

    # Draw relations

    # Filter out relations
    ### Debug: output all relations if not enough
    n_rel_topk = min(num_rel, len(result.labels)//2)
    # Exclude background class
    rel_dists = result.rel_dists[:, 1:]
    # rel_dists = result.rel_dists
    rel_scores = rel_dists.max(1)
    # rel_scores = result.triplet_scores
    # Extract relations with top scores
    rel_topk_idx = np.argpartition(rel_scores, -n_rel_topk)[-n_rel_topk:]
    rel_labels_topk = rel_dists[rel_topk_idx].argmax(1)
    rel_pair_idxes_topk = result.rel_pair_idxes[rel_topk_idx]
    relations = np.concatenate(
        [rel_pair_idxes_topk, rel_labels_topk[..., None]], axis=1)
    n_rels = len(relations)
    
    # top_padding = 20
    # bottom_padding = 20
    # left_padding = 20
    # text_size = 10
    # text_padding = 5
    # text_height = text_size + 2 * text_padding
    # row_padding = 10
    # height = (top_padding + bottom_padding + n_rels *
    #           (text_height + row_padding) - row_padding)
    # width = img_w
    # curr_x = left_padding
    # curr_y = top_padding
    
    # # # Adjust colormaps
    # # colormap_coco = [adjust_text_color(c, viz) for c in colormap_coco]
    # viz_graph = VisImage(np.full((height, width, 3), 255))
    
    all_rel_vis = []
    
    for i, r in enumerate(relations):
        s_idx, o_idx, rel_id = r
        s_label = rel_obj_labels[s_idx] # subject (e.g. person)
        o_label = rel_obj_labels[o_idx] # target object? (e.g. spoon)
        rel_label = PREDICATES[rel_id] # relation (e.g. holding).. so person holding spoon
        all_rel_vis.append([s_label, rel_label, o_label]) # BK
        # viz = Visualizer(img)
        # viz.overlay_instances(
        #     labels=[s_label, o_label],
        #     masks=[masks[s_idx], masks[o_idx]],
        #     assigned_colors=[colormap_coco[s_idx], colormap_coco[o_idx]],
        # )
        # viz_masked_img = viz.get_output().get_image()

        # viz_graph = VisImage(np.full((40, width, 3), 255))
        # curr_x = 2
        # curr_y = 2
        # text_size = 25
        # text_padding = 20
        # font = 36
        # text_width = draw_text(
        #     viz_img=viz_graph,
        #     text=s_label,
        #     x=curr_x,
        #     y=curr_y,
        #     color=colormap_coco[s_idx],
        #     size=text_size,
        #     padding=text_padding,
        #     font=font,
        # )
        # curr_x += text_width
        # # Draw relation text
        # text_width = draw_text(
        #     viz_img=viz_graph,
        #     text=rel_label,
        #     x=curr_x,
        #     y=curr_y,
        #     size=text_size,
        #     padding=text_padding,
        #     box_color='gainsboro',
        #     font=font,
        # )
        # curr_x += text_width

        # # Draw object text
        # text_width = draw_text(
        #     viz_img=viz_graph,
        #     text=o_label,
        #     x=curr_x,
        #     y=curr_y,
        #     color=colormap_coco[o_idx],
        #     size=text_size,
        #     padding=text_padding,
        #     font=font,
        # )
        # output_viz_graph = np.vstack([viz_masked_img, viz_graph.get_image()])
        # if show:
        #    all_rel_vis.append(output_viz_graph)
        
        # mmcv.imwrite(output_viz_graph, '{}_relations.jpg'.format(i))

    # return all_rel_vis
    
    # BK
    return all_rel_vis


def make_gif(np_images):
    frames = [Image.fromarray(numpy_image.astype('uint8'), 'RGB') for numpy_image in np_images]
    # frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    file_name = "top_rel.gif"
    frame_one.save(file_name, format="GIF", append_images=frames,
               save_all=True, duration=1000, loop=0)
    return file_name