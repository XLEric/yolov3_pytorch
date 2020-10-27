#coding:utf-8
# date:2019-08
# Author: X.li
# function: predict
import argparse
import time
import os
import torch
from utils.datasets import *
from utils.utils import *
from utils.parse_config import parse_data_cfg
from yolov3 import Yolov3, Yolov3Tiny
from utils.torch_utils import select_device
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# cuda = torch.cuda.is_available()
# device = torch.device('cuda:0' if cuda else 'cpu')

def process_data(img, img_size=416):# 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img

def show_model_param(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        print("该层的结构: {}, 参数和: {}".format(str(list(i.size())), str(l)))
        k = k + l
    print("----------------------")
    print("总参数数量和: " + str(k))

def detect(
        model_path,
        root_path,
        cfg,
        data_cfg,
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
):
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    num_classes = len(classes)
    # Initialize model
    if "-tiny" in cfg:
        model = Yolov3Tiny(num_classes)
        weights = model_path
    else:
        model = Yolov3(num_classes)
        weights = model_path

    show_model_param(model)# 显示模型参数

    device = select_device() # 运行硬件选择
    use_cuda = torch.cuda.is_available()
    # Load weights
    if os.access(weights,os.F_OK):# 判断模型文件是否存在
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        print('error model not exists')
        return False
    model.to(device).eval()#模型设置为 eval

    colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, num_classes + 1)][::-1]

    for img_name in os.listdir(root_path):
        img_path  = root_path + img_name
        im0 = cv2.imread(img_path)
        print("---------------------")

        t = time.time()
        img = process_data(im0, img_size)
        print('img shape   @@@@@@@@@@@@@@@@ ',img.shape)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print("process time:", t1-t)
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        pred, _ = model(img)#图片检测
        if use_cuda:
            torch.cuda.synchronize()
        t2 = time.time()
        print("inference time:", t2-t1)
        detections = non_max_suppression(pred, conf_thres, nms_thres)[0] # nms
        if use_cuda:
            torch.cuda.synchronize()
        t3 = time.time()
        print("get res time:", t3-t2)
        if detections is None or len(detections) == 0:
            continue
        # Rescale boxes from 416 to true image size
        detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape).round()
        result = []
        for res in detections:
            result.append((classes[int(res[-1])], float(res[4]), [int(res[0]), int(res[1]), int(res[2]), int(res[3])]))
        if use_cuda:
            torch.cuda.synchronize()
        s2 = time.time()
        print("detect time:", s2 - t)
        print(result)

        # Draw bounding boxes and labels of detections
        for *xyxy, conf, cls_conf, cls in detections:
            label = '%s %.2f' % (classes[int(cls)], conf)
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        cv2.namedWindow('result',0)
        cv2.imshow("result", im0)
        key = cv2.waitKey(0)
        if key == 27:
            break

if __name__ == '__main__':


    model_path = './coco_model/yolov3_coco.pt' # 检测模型路径
    root_path = './test_images/'# 测试文件夹
    model_cfg = 'yolov3' # 模型类型
    voc_config = 'cfg/voc.data' # 模型相关配置文件
    img_size = 416 # 图像尺寸
    conf_thres = 0.35# 检测置信度
    nms_thres = 0.5 # nms 阈值

    with torch.no_grad():#设置无梯度运行
        detect(
            model_path = model_path,
            root_path = root_path,
            cfg = model_cfg,
            data_cfg = voc_config,
            img_size=img_size,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
        )
