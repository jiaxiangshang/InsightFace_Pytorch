# -*- coding: utf-8 -*-
# @Author  : Jiaxiang Shang
# @Email   : jiaxiang.shang@gmail.com
# @Time    : 9/29/20 3:40 PM

import os, sys
import argparse
from shutil import copyfile
#
import numpy as np
#import pywavefront, trimesh
import skimage.io as skio
import cv2

# arch face
import cv2
from PIL import Image
import argparse
#from pathlib import Path
from multiprocessing import Process,Pipe,Value,Array
import torch

# self
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./
_tf_dir = os.path.dirname(_cur_dir) # ./
_tool_data_dir = os.path.dirname(_tf_dir) # ../
_deep_learning_dir = os.path.dirname(_tool_data_dir) # ../
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir) # /home/..../pytorch3d

from thirdParty.InsightFace_Pytorch.config import get_config
from thirdParty.InsightFace_Pytorch.mtcnn import MTCNN
from thirdParty.InsightFace_Pytorch.Learner import face_learner
from thirdParty.InsightFace_Pytorch.utils import load_facebank, draw_box_name, prepare_facebank

from baselib_python.IO.Landmark import *

parser = argparse.ArgumentParser(description='Preprocess Altizure Pipline')

# 0.
parser.add_argument('--dic_dataset', type=str, default='/data0/0_DATA/0_Face_3D/0_facescape/14_fs_glist_miniHW_alignNonRigid', help='')
parser.add_argument('--gl_name_split', type=str, default='train', help='')
parser.add_argument('--path_3dmm_top', type=str, default='/data0/0_DATA/0_Face_3D/0_facescape/0_facescape_model', help='')

PARSER = parser.parse_args()

def test_model():
    work_path = os.path.join(_cur_dir, 'work_space')
    conf = get_config(False, work_path)

    # mtcnn
    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)
    learner.threshold = 1.54
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'ir_se50.pth', False, True)
    else:
        learner.load_state(conf, 'ir_se50.pth', False, True)
    learner.model.eval()
    return mtcnn, learner, conf


def test_archface(mtcnn, learner, conf, path_image, lm2d_5=None):
    if lm2d_5 is not None:
        if isinstance(lm2d_5, str):
            lm2d_5 = parse_self_lm(lm2d_5)

    print('learner loaded')
    with torch.no_grad():
        # Main loop
        dic_image, name_image = os.path.split(path_image)
        name_pure, _ = os.path.splitext(name_image)

        # texture
        image_bgr = cv2.imread(path_image)
        # image = image_bgr[..., ::-1]
        image = Image.fromarray(image_bgr)
        #
        image_align = mtcnn.align(image, lm2d_5)
        if image_align is not None:
            if 0:
                cv2.imshow("Image", np.asarray(image))
                k = cv2.waitKey(0) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                cv2.imshow("Image Align", np.asarray(image_align))
                k = cv2.waitKey(0) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
            results_512 = learner.infer_feature(conf, image_align)
            path_results_512_save = os.path.join(dic_image, name_pure + '_id_arcface.npy')
            np.save(path_results_512_save, results_512.detach().cpu().numpy())

if __name__ == "__main__":
    #
    pass