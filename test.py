import os.path
import logging

import numpy as np
from collections import OrderedDict
import cv2

import time
import torch
from io import BytesIO
from PIL import Image

import util
from network import NITRE as net

def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    L_path = './test/001/' 
    E_path = './test_output/001/'
    model_path = 'nitre.pth'
    util.mkdir(E_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    model = net()
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
   
    # ----------------------------------------
    # test
    # ----------------------------------------
    L_paths = util.get_image_paths(L_path)
    cost_time,cnt = 0,0
    for idx, img in enumerate(L_paths):
        if 'checkpoint' in img: continue
        img_name = os.path.basename(img)
        print(idx, img)
        
        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)
        
        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        start = time.time()
        img_E = model(img_L)
        end = time.time()
        cost_time += (end-start)
        cnt += 1
        
        img_E = util.tensor2uint(img_E)

        # ------------------------------------
        # save results
        # ------------------------------------
        util.imsave(img_E, os.path.join(E_path, img_name))
        
    print("Overall runtime:{}, images num:{}, runtime per image:{}".format(cost_time, cnt, cost_time/cnt))
        
        
if __name__ == '__main__':
    main()
