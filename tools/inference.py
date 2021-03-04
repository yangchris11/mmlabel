from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet import datasets

import numpy as np

import tqdm

import pycocotools.mask as cocomask

video_basename = 'CAM10-2021-0224-135954-140054'

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

# groundtruth path to read
gt_filename = 'gt_{}.txt'.format(video_basename)
f = open(gt_filename, 'w')

# test a video and show the results
video_filename = '../data/{}.mp4'.format(video_basename)

video = mmcv.VideoReader(video_filename)


# id map
idmap = {}


print('Using config file: {}'.format(config_file))
print('      ckpt file: {}'.format(checkpoint_file))
print('      gt path: {}'.format(gt_filename))
print('      video path: {}'.format(video_filename))

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
    
    # original image will be store in /msc/{videoname}/img/xxxxxx.jpg
    model.show_result(frame, result, score_thr=2, out_file='../msc/{}/img/{:06d}.jpg'.format(video_basename ,i+1))
    # detection result will be store in /msc/{videoname}/det/xxxxxx.jpg
    model.show_result(frame, result_person_only, score_thr=0.9, bbox_color='red',text_color='red',out_file='../msc/{}/det/{:06d}.jpg'.format(video_basename, i+1))

    