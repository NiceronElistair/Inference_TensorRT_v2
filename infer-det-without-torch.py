import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import time

from config import CLASSES, COLORS
from models_v2.utils import blob, letterbox, draw_bounding_boxes, scale_boxes, non_max_suppression, NAME_CLASSES

def main(args: argparse.Namespace) -> None:
    
    from models_v2.pycuda_api import TRTEngine
    conf_thres = 0.5
    iou_thres = 0.5

    Engine = TRTEngine(args.engine)
    H, W = Engine.inp_info[0].shape[-2:]

    cap = cv2.VideoCapture(0)

    prev_frame_time = 0
    new_frame_time = 0

    while True:

        _, bgr = cap.read()
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)

        new_frame_time = time.time()


        # inference
        data = Engine(tensor)

        prediction = data[-1]
        prediction = torch.Tensor(prediction).to('cuda')
        
        y = non_max_suppression(prediction, conf_thres, iou_thres)        

        det = y[0] # list of list to list
        det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], draw.shape).round() # bounding box coord are not adapted to the im0 frame, need to rescale the coord
        
        draw = draw_bounding_boxes(det, draw) # draw boundinx box dans print label name and confidence score

        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)


        bottomLeftCornerOfText = (450, 50)
        cv2.putText(draw, f'FPS: {fps}', bottomLeftCornerOfText,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, 2) # print frame rate on frame
        cv2.imshow('Frame', draw)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # destroy window
            break

    cap.release()
    cv2.destroyAllWindows()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--method',
                        type=str,
                        default='cudart',
                        help='CUDART pipeline')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
