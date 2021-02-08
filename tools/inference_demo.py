from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet import datasets

import numpy as np

import tqdm

import pycocotools.mask as cocomask



def get_mask_IOU(mask1, mask2):
    #print(mask1)
    ratio = cocomask.iou([mask1], [mask2], [0])[0][0]
    #ratio2=  rletools.area(rletools.merge([mask1, mask2], intersect=True))/rletools.area(rletools.merge([mask1, mask2]))
    #if ratio!=ratio2:
    #    import pdb;pdb.set_trace()
    ratio2 = cocomask.area(cocomask.merge([mask1, mask2], intersect=True)) / min(cocomask.area(mask1),cocomask.area(mask2))
    return ratio2

# Specify the path to model config and checkpoint file
config_file = '../configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco.py'
checkpoint_file = '../checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201-0f411b1f.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# write to groundtruth file
f = open('gt_cam4.txt', 'w')

# test a video and show the results
video = mmcv.VideoReader('../data/cam4.mp4')


# id map
idmap = {}

for i, frame in enumerate(tqdm.tqdm(video)):
    result = inference_detector(model, frame)
    # person is first class in COCO
    # so extract person bbox and mask from the entire result
    result_person_only = ([result[0][0]], [result[1][0]])
    result_bbox = [result[0][0]]
    result_mask = [result[1][0]]

    # gt.txt structure
    # clsid = 1 we set the clsid to objid for easier labeling
    # frameid, objid, clsid, img_h, img_w, rle_code
    start = 0
    
    pairs = sorted(zip(result_bbox[0], result_mask[0]), key = lambda x: x[0][0]) 



    for bbox, mask in pairs:
        if bbox[-1] >= 0.9:
            rle_code = cocomask.encode(np.asfortranarray(mask))
            f.write('{} {} {} {} {} {}\n'.format(
                i+1,
                start,
                1,
                720, 1280,
                rle_code['counts'].decode('ascii')
            ))
            start += 1
        
    model.show_result(frame, result, score_thr=1.1, out_file='../msc/cam4/img/{:06d}.jpg'.format(i+1))
    model.show_result(frame, result_person_only, score_thr=0.9, bbox_color='red',text_color='red',out_file='../msc/cam4/det/{:06d}.jpg'.format(i+1))

    