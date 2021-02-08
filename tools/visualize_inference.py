from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet import datasets

import numpy as np

import pycocotools.mask as cocomask


def get_bbox(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    return bbox


# video name
video_basename = 'cam1'

# Specify the path to model config and checkpoint file
config_file = '../configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco.py'
checkpoint_file = '../checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201-0f411b1f.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# groundtruth path to read
# gt_filename = 'gt_{}.txt'.format(video_basename)
gt_filename = 'cam_v1_001.txt'
f = open(gt_filename, 'r')

video_filename = '../data/{}.mp4'.format(video_basename)

video = mmcv.VideoReader(video_filename)
bbox_frame = {}
mask_frame = {}


for line in f.readlines():


    frameid, _, xmin, ymin, xmax, ymax, conf, img_h, img_w, rle_code = line.split()
    img_w = int(img_w)
    img_h = int(img_h)
    rle = {"size": [img_h, img_w], "counts": rle_code}
    mask = cocomask.decode(rle)
    frameid = int(frameid)
    # objid = int(objid)
    # clsid = int(clsid)

    if float(conf) < 0.5:
        continue

    ymin, xmin, ymax, xmax = get_bbox(mask)

    if frameid not in bbox_frame:
        objid = 0
        bbox_frame[frameid] = [np.zeros((0, 5)) for _ in range(10)]
        mask_frame[frameid] = [[] for _ in range(10)]
        # bbox_frame[frameid][0] = []#
 

    bbox_frame[frameid][objid] = np.asarray([[xmin, ymin, xmax, ymax, float(conf)]])
    mask_frame[frameid][objid] = np.asarray([mask])
    objid += 1
    # print(bbox_frame[frameid])


print(bbox_frame[1])
        
model.CLASSES = ('people', 'people', 'people', 'people', 'people', 'people', 'people', 'people', 'people', 'people')



# bbox_frame = np.asarray(bbox_frame)
# mask_frame = np.asarray(mask_frame)

import tqdm

for i, frame in enumerate(tqdm.tqdm(video)):
    if i+1 not in bbox_frame:
        continue
    model.show_result(frame, (bbox_frame[i+1], mask_frame[i+1]),bbox_color='red',text_color='red',out_file='../msc/{}/inferall/{:06d}.jpg'.format(video_basename, i+1))
