import sys
sys.path.append('.')
sys.path.append('..')
import matplotlib.pyplot as plt
from utils.uvd_transform import uvdtouvd
import os
from os.path import join
import cv2 as cv
import json
import numpy as np
import torch
from PIL import Image
from IPython.display import Image as Img
from IPython.display import display
import glob

baseDir = './test_tw'
handposeRoot = join(baseDir, 'onlyHandWorldCoordinate_uvd.json')

def visualize(refined_joint, frame, refcam, Ks, ext, idx):
    init = True
    if init == True:
        with open(handposeRoot, 'r') as fi:
                data = json.load(fi)
        handpose_0 = np.array(data['0_%d'%refcam])
        handpose_1 = np.array(data['1_%d'%refcam])
        handpose_2 = np.array(data['2_%d'%refcam]) #[116, 21, 2]
        init_joint = torch.FloatTensor(np.mean([handpose_0, handpose_1, handpose_2], axis=0))[frame]
        init = False
        fig = plt.figure(figsize=(15, 5))
        ax0 = fig.add_subplot(131) #0
        ax1 = fig.add_subplot(132) #1
        ax2 = fig.add_subplot(133) #2
        joint = {}
        for i in [0,1,2]:
            if i == refcam:
                joint['cam%d'%i] = init_joint
            else:
                joint['cam%d'%i] = uvdtouvd(init_joint, Ks[refcam], Ks[i], ext['ref%d'%refcam][i])
        
        img_0 = cv.imread('./self/record_multi/220405_hand/rgb/mas_%d.png'%(30+frame))
        img_1 = cv.imread('./self/record_multi/220405_hand/rgb/sub1_%d.png'%(30+frame))
        img_2 = cv.imread('./self/record_multi/220405_hand/rgb/sub2_%d.png'%(30+frame))
        img_0 = cv.cvtColor(img_0, cv.COLOR_BGR2RGB)
        img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
        img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)
        ax0.imshow(img_0)
        ax1.imshow(img_1)
        ax2.imshow(img_2)
        ax0.scatter(joint['cam0'][:,0], joint['cam0'][:,1], s=1)
        ax1.scatter(joint['cam1'][:,0], joint['cam1'][:,1], s=1)
        ax2.scatter(joint['cam2'][:,0], joint['cam2'][:,1], s=1)
        fig.savefig('./results/0.png')
        plt.close(fig)
        

    fig = plt.figure(figsize=(15, 5))
    ax0 = fig.add_subplot(131) #0
    ax1 = fig.add_subplot(132) #1
    ax2 = fig.add_subplot(133) #2
    joint = {}
    for i in [0,1,2]:
        if i == refcam:
            joint['cam%d'%i] = refined_joint.detach().cpu().numpy()
        else:
            joint['cam%d'%i] = uvdtouvd(refined_joint, Ks[refcam], Ks[i], ext['ref%d'%refcam][i]).detach().cpu().numpy()
    
    img_0 = cv.imread('./self/record_multi/220405_hand/rgb/mas_%d.png'%(30+frame))
    img_1 = cv.imread('./self/record_multi/220405_hand/rgb/sub1_%d.png'%(30+frame))
    img_2 = cv.imread('./self/record_multi/220405_hand/rgb/sub2_%d.png'%(30+frame))
    img_0 = cv.cvtColor(img_0, cv.COLOR_BGR2RGB)
    img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
    img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)
    ax0.imshow(img_0)
    ax1.imshow(img_1)
    ax2.imshow(img_2)
    ax0.scatter(joint['cam0'][:,0], joint['cam0'][:,1], s=1)
    ax1.scatter(joint['cam1'][:,0], joint['cam1'][:,1], s=1)
    ax2.scatter(joint['cam2'][:,0], joint['cam2'][:,1], s=1)
    fig.savefig('./results/%d.png'%(idx+1))
    plt.close(fig)
