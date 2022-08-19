import sys
import os
from os.path import join as join
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)), './optimization'))

import pickle
import multiprocessing as mlp
from enum import IntEnum
import numpy as np
from random import shuffle
from absl import app

from eval import utilsEval
from lift2DJoints import lift2Dto3DMultiView
import manoHandVis as manoVis
import json
import cv2


baseDir = './self'
handposeRoot = join(baseDir, 'predictedPose', 'onlyHandWorldCoordinate_uvd.json')
recordDir = join(baseDir, 'record_multi', '220405_hand')


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def getPredicedPoses():
    # data : (cam_idx, frame_idx, joint_idx, 3) ~ (3, 116, 21, 3)
    #data = np.load(handposeRoot)
    _assert_exist(handposeRoot)
    with open(handposeRoot, 'r') as fi:
        data = json.load(fi)
    return data


def readRecord(idx, mode, init_frame=30):
    # rgb : recordDir + '/rgb/' + ['mas_%.png', 'sub1_%.png', 'sub2_%.png']
    # depth : recordDir + '/depth/' + ['mas_%.png', 'sub1_%.png', 'sub2_%.png']
    idx += init_frame
    if mode is 'rgb':
        rgb_mas_path = join(recordDir, 'rgb/') + 'mas_' + str(idx) + '.png'
        rgb_sub1_path = join(recordDir, 'rgb/') + 'sub1_' + str(idx) + '.png'
        rgb_sub2_path = join(recordDir, 'rgb/') + 'sub2_' + str(idx) + '.png'

        rgb_mas = cv2.imread(rgb_mas_path)
        rgb_sub1 = cv2.imread(rgb_sub1_path)
        rgb_sub2 = cv2.imread(rgb_sub2_path)

        img_set = [rgb_mas, rgb_sub1, rgb_sub2]

    if mode is 'depth':
        assert 'Not implemented'

    return img_set


def getMultiviewPose(dummy, MVHandposeSet, ownrefhand_set, refIdx, startIdx, saveMVFitDir, ref_idx):
    num_views = len(MVHandposeSet)
    seq_len = len(MVHandposeSet[0])
    for idx in range(seq_len):
        MVHandpose = MVHandposeSet[:, idx]
        ownrefhand = ownrefhand_set[:, idx]
        initPose = np.mean(MVHandpose, axis=0)

        # set larger weight on reference cam's prediction
        weight = np.ones((num_views, 1), dtype=np.float32) / 2.0
        weight[ref_idx] = 1.0
        outputPose = lift2Dto3DMultiView(MVHandpose, initPose, ownrefhand, refIdx, weights=weight, outDir=saveMVFitDir)
        curr_idx = int(startIdx+idx)
        newDict = {'sourceKPS': MVHandpose,
                   'fittedKPS': outputPose,
                   'idx': curr_idx}

        with open(join(saveMVFitDir, str(curr_idx) + '.pickle'), 'wb') as f:
            pickle.dump(newDict, f)


def main(argv):
    ### initialize
    saveMVFitDir = join(baseDir, 'multiviewFit_ref2')
    if not os.path.exists(saveMVFitDir):
        os.mkdir(saveMVFitDir)

    # let sub2 as reference camera
    cam_nums = 3
    ref_idx = 2

    source = getPredicedPoses()
    handpose_0 = np.array(source['0_%d'%ref_idx])
    handpose_1 = np.array(source['1_%d'%ref_idx])
    handpose_2 = np.array(source['2_%d'%ref_idx])

    handpose_set = np.array([handpose_0, handpose_1, handpose_2])

    ownrefhand_0 = np.array(source['0_0'])
    ownrefhand_1 = np.array(source['1_1'])
    ownrefhand_2 = np.array(source['2_2'])

    ownrefhand_set = np.array([ownrefhand_0, ownrefhand_1, ownrefhand_2])

    print("starting multi-view optimization...")
    numThreads = 1
    numCandidateFrames = len(handpose_set[0])
    numFramesPerThread = np.ceil(numCandidateFrames / numThreads).astype(np.uint32)
    procs = []
    for proc_index in range(numThreads):
        startIdx = proc_index * numFramesPerThread
        endIdx = min(startIdx + numFramesPerThread, numCandidateFrames)
        args = ([], handpose_set[:, startIdx:endIdx], ownrefhand_set[:, startIdx:endIdx], ref_idx, startIdx, saveMVFitDir, ref_idx)
        proc = mlp.Process(target=getMultiviewPose, args=args)

        proc.start()
        procs.append(proc)
    for i in range(len(procs)):
        procs[i].join()
    print("Done")

    # read the new pickle files created after multi-view fitting
    perFrameFittingDataList = []
    for idx in range(numCandidateFrames):
        assert os.path.exists(
            join(saveMVFitDir, str(idx) + '.pickle')), 'Multi-view fitting failed!'
        with open(join(saveMVFitDir, str(idx) + '.pickle'), 'rb') as f:
            pklData = pickle.load(f)
        perFrameFittingDataList.append(pklData)

    sourceKPS_np = np.stack([pklData['sourceKPS'] for pklData in perFrameFittingDataList], axis=0)
    fittedKPS_np = np.stack([pklData['fittedKPS'] for pklData in perFrameFittingDataList], axis=0)
    idx_np = np.stack([pklData['idx'] for pklData in perFrameFittingDataList], axis=0)

    # visualize fitted output
    width = int(1280 / 2.)
    height = int(720 / 2.)
    for idx in range(numCandidateFrames):
        img_set = readRecord(idx, mode='rgb')
        img_ref = cv2.resize(img_set[ref_idx], (width, height)) / 255.0

        fitted_pose = np.copy(fittedKPS_np[idx]).astype(np.float32)
        fitted_pose[:, :-1] /= 2.0

        img_cam = manoVis.showHandJoints_cv(img_ref.copy(), fitted_pose)
        cv2.imshow("cam_ref", img_cam)
        print("index : ", idx)
        cv2.waitKey(20)

    print("end")


if __name__ == '__main__':
    app.run(main)