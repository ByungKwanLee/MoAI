import torch
import numpy as np
from detectron2.structures import BitMasks
from utils.constants import COCO_PANOPTIC_CLASSES, PREDICATES, ADE20K_847, IMAGENET_CLASSES

def make_system_prompt(processor, device, ignore_index, img_length=1225):
    # system prompt
    system_prompt = make_human_string("AI assistant should give helpful and detailed answers to user after fully understanding an image.",
                                    "<image>")

    length = processor(system_prompt, return_tensors='pt').input_ids[0].shape[0]
    moai_label = torch.tensor([ignore_index]*(length+img_length-1)).to(device)
    im_mask = torch.zeros_like(moai_label)
    im_mask[-img_length:]=1

    return system_prompt, moai_label, im_mask

def demo_make_and_add_prompt_and_im_mask(moai_prompt, im_mask, prompt, processor, device):
    
    # indent
    prompt = " USER: " + prompt + " ASSISTANT:"

    # input_ids and 
    label_ids = processor(prompt, return_tensors='pt', add_special_tokens=False).input_ids[0]
    
    # Concat previous prompt + current prompt
    moai_prompt += prompt
    im_mask = torch.tensor(im_mask.tolist() + torch.zeros_like(label_ids).tolist()).to(device)
    
    return moai_prompt, im_mask
    
def make_and_add_prompt_and_im_mask(moai_prompt, im_mask, prompt, processor, device):
    
    # indent
    prompt = " [UNUSED_TOKEN_146]user\n" + prompt + "[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"

    # input_ids and 
    label_ids = processor(prompt, return_tensors='pt', add_special_tokens=False).input_ids[0]
    
    # Concat previous prompt + current prompt
    moai_prompt += prompt
    im_mask = torch.tensor(im_mask.tolist() + torch.zeros_like(label_ids).tolist()).to(device)
    
    return moai_prompt, im_mask

def make_human_string(*args):
    out = ''
    for ind, arg in enumerate(args):
        out += arg
        if len(args)-1 != ind: out += ' '
    return out

def box_and_class_parser(decoded_text):
    start_box_index = find(decoded_text, '[')
    end_box_index = find(decoded_text, ']')

    start_class_index = find(decoded_text, '(')
    end_class_index = find(decoded_text, ')')
    
    if len(start_box_index) != len(end_box_index): return None, None, True
    if len(start_class_index) != len(end_class_index): return None, None, True
    if len(start_class_index) != len(start_box_index): return None, None, True

    box_list = []
    class_list = []
    for sb, eb, sc, ec in zip(start_box_index, end_box_index, start_class_index, end_class_index):
        box_list.append(eval(decoded_text[sb: eb+1]))
        class_list.append(decoded_text[sc+1: ec][decoded_text[sc+1: ec].find(' ')+1:])
        if len(box_list[-1]) != 4: box_list.pop(-1); class_list.pop(-1)
    box_tensor = torch.tensor(box_list)
    return box_tensor, class_list, False

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def list2string(_list):
    out = ''
    for i, x in enumerate(_list):
        out+=str(x)
        if i!=len(_list)-1: out+=', '
    out += ''
    return out

def box2string(box):
    out = '['
    for i, x in enumerate(box):
        out+=f"{round(x.item(), 2):.2f}"
        if i!=len(box)-1: out+=', '
    out += ']'
    return out

def boxes2string(boxes):
    out = ''
    for i, x in enumerate(boxes):
        out+=box2string(x)
        if i!=len(boxes)-1: out+=', '
    out += ''
    return out

def classescolors2string(nice_seg_info_list):
    classes = [nice_seg_info['class'] for nice_seg_info in nice_seg_info_list]
    colors = [nice_seg_info['color'] for nice_seg_info in nice_seg_info_list]

    count = {}
    out = ''
    for i, (x, y) in enumerate(zip(classes, colors)):
        if x in count.keys():
            count[x]+=1
        else:
            count.update({x: 1})
        out+=f"({count[x]} {x}) {y}"
        if i!=len(classes)-1: out+=', '
    return out


def classesboxes2string(nice_seg_info_list, class_name='all'):
    if class_name == 'all':
        classes = [nice_seg_info['class'] for nice_seg_info in nice_seg_info_list]
        boxes = [nice_seg_info['box'] for nice_seg_info in nice_seg_info_list]
    else:
        classes = [nice_seg_info['class'] for nice_seg_info in nice_seg_info_list if nice_seg_info['class']==class_name]
        boxes = [nice_seg_info['box'] for nice_seg_info in nice_seg_info_list if nice_seg_info['class']==class_name]

    count = {}
    out = ''
    for i, (x, y) in enumerate(zip(classes, boxes)):
        if x in count.keys():
            count[x]+=1
        else:
            count.update({x: 1})
        out+=f"(#{count[x]} {x}) {box2string(y)}"
        if i!=len(classes)-1: out+=', '
    return out

def classes2string(nice_seg_info_list):
    classes = [nice_seg_info['class'] for nice_seg_info in nice_seg_info_list]
    count = {}
    out = ''
    for i, x in enumerate(classes):
        if x in count.keys():
            count[x]+=1
        else:
            count.update({x: 1})
        out+=f"#{count[x]} {x}"
        if i!=len(classes)-1: out+=', '
    return out


def create_pascal_label_colormap(num=1+133):
    def bit_get(val, idx):
        return (val >> idx) & 1
    colormap = np.zeros((num, 3), dtype=int)
    ind = np.arange(num, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3
    return colormap / 255

def make_seg_prompt(seg_result):
    seg_ind = seg_result['segmentation'].clone().to(torch.int32)
    seg_info = seg_result['segments_info']
    
    seg_ind[torch.where(seg_ind<=0)] = 0 # Filtering unknown and background

    # Generating Color Map
    cmap = create_pascal_label_colormap()
    panoptic_map = cmap[seg_ind.cpu()]

    # Generating Boxes
    try:
        boxes = BitMasks(torch.stack([seg_ind == i+1 for i in range(seg_ind.max()) if (seg_ind == i+1).sum() != 0])).get_bounding_boxes()
        boxes.scale(1/490, 1/490)

        # Panoptic Index to seg info including box
        nice_seg_info_list = []
        for i, e in enumerate(seg_info):
            nice_seg_info_list.append(
            {'id': e['id'],
            'class': COCO_PANOPTIC_CLASSES[e['label_id']].replace('-merged','').replace('-other','').replace('-stuff',''),
            'box': boxes.tensor[i]
                })
    except:
        nice_seg_info_list= []
    
    # Verbalization
    if len(nice_seg_info_list)!=0:
        verbalization_seg = 'The image includes bounding box coordinates and their objects: '
        for i, nice_ele in enumerate(nice_seg_info_list):
            box = box2string(nice_ele['box'])
            cls = nice_ele['class']
            verbalization_seg += f'{box} {cls}'
            if i!=len(nice_seg_info_list)-1: verbalization_seg += ', and '
        verbalization_seg += '.'
    else:
        verbalization_seg = ''
    
    return verbalization_seg, panoptic_map, nice_seg_info_list

def make_od_prompt(od_result):
    od_scores = od_result['scores']
    od_index = torch.where(od_scores>=0.5)
    od_boxes = od_result['boxes'][od_index] / 490
    od_labels = [(ADE20K_847+IMAGENET_CLASSES)[ind] for ind in od_result['labels'][od_index]]
    
    if len(od_boxes)!=0:
        verbalization_od='The image includes bounding box coordinates and their objects: '
        for i, (box, label) in enumerate(zip(od_boxes, od_labels)):
            verbalization_od += f'{box2string(box)} {label}'
            if i!=len(od_boxes)-1: verbalization_od += ', and '
        verbalization_od +='.'
    else:
        verbalization_od=''
    return verbalization_od, od_boxes, od_labels
    

def make_sgg_prompt(sgg_result, num_rel=10):
    from mmdet.evaluation import INSTANCE_OFFSET
    
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
    
    # Draw masks
    pan_results = sgg_result.pan_results

    ids = np.unique(pan_results)[::-1]
    num_classes = 133
    legal_indices = (ids != num_classes)  # for VOID label
    ids = ids[legal_indices]

    # Get predicted labels
    labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    labels = [COCO_PANOPTIC_CLASSES[l] for l in labels] # what objects are in the image

    #For psgtr
    rel_obj_labels = sgg_result.labels
    rel_obj_labels = [COCO_PANOPTIC_CLASSES[l - 1].replace('-merged','').replace('-other','').replace('-stuff','') for l in rel_obj_labels] # what objects are in the image
    rel_obj_labels = label_duplicates(rel_obj_labels)
   
    try:
        # Filter out relations
        ### Debug: output all relations if not enough
        n_rel_topk = min(num_rel, len(rel_obj_labels))
        # Exclude background class
        rel_dists = sgg_result.rel_dists[:, 1:]
        rel_scores = rel_dists.max(1)
        # Extract relations with top scores
        rel_topk_idx = np.argpartition(rel_scores, -n_rel_topk)[-n_rel_topk:]
        sgg_threshold = np.partition(rel_scores, -n_rel_topk)[-n_rel_topk:]
        rel_labels_topk = rel_dists[rel_topk_idx[np.where(sgg_threshold>=0.8)]].argmax(1)
        rel_pair_idxes_topk = sgg_result.rel_pair_idxes[rel_topk_idx[np.where(sgg_threshold>=0.8)]]
        relations = np.concatenate(
            [rel_pair_idxes_topk, rel_labels_topk[..., None]], axis=1)

        # Verbalization
        if len(relations)!=0:
            verbalization_sgg = 'The image includes relationships between objects: '
            for i, r in enumerate(relations):
                s_idx, o_idx, rel_id = r
                s_label = rel_obj_labels[s_idx] # subject (e.g. person)
                o_label = rel_obj_labels[o_idx] # target object? (e.g. spoon)
                rel_label = PREDICATES[rel_id] # relation (e.g. holding).. so person holding spoon
                verbalization_sgg += (f'{s_label} is {rel_label} {o_label}') # BK
                if i!=len(relations)-1:
                    verbalization_sgg += ', and '
            verbalization_sgg += '.'
        else:
            verbalization_sgg = ''        

    except:
        verbalization_sgg = ''
        
    return verbalization_sgg

def make_ocr_prompt(ocr_result):
    ocr_texts = [text_inform[1][0] for text_inform in ocr_result if text_inform[1][1] >= 0.85]
    
    # Verbalization
    if len(ocr_texts)!=0:
        verbalization_ocr = 'The image includes text descriptions: '
        for i, txt in enumerate(ocr_texts):
            verbalization_ocr += txt
            if i!=len(ocr_texts)-1:
                verbalization_ocr += ', and '
        verbalization_ocr += '.'
    else:
        verbalization_ocr = ''
    return verbalization_ocr