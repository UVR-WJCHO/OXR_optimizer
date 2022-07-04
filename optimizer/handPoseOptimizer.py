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


baseDir = './test_seq/self'
handposeRoot = join(baseDir, 'predictedPose', 'onlyHandWorldCoordinate.npy')
recordDir = join(baseDir, 'record_multi', '220405_hand')


def getPredicedPoses():
    # data : (cam_idx, frame_idx, joint_idx, 3) ~ (3, 116, 21, 3)
    data = np.load(handposeRoot)
    return data


def readRecord(idx):
    # rgb : recordDir + '/rgb/' + ['mas_%.png', 'sub1_%.png', 'sub2_%.png']
    # depth : recordDir + '/depth/' + ['mas_%.png', 'sub1_%.png', 'sub2_%.png']
    return None


def getMultiviewPose(dummy, MVHandposeSet, startIdx, saveMVFitDir):
    num_views = len(MVHandposeSet)
    seq_len = len(MVHandposeSet[0])

    for idx in range(seq_len):
        MVHandpose = MVHandposeSet[:, idx]
        initPose = np.mean(MVHandpose, axis=0)

        outputPose = lift2Dto3DMultiView(MVHandpose, initPose, weights=np.ones((num_views, 1), dtype=np.float32), outDir=saveMVFitDir)
        curr_idx = int(startIdx+idx)
        newDict = {'sourceKPS': MVHandpose,
                   'fittedKPS': outputPose,
                   'idx': curr_idx}

        with open(join(saveMVFitDir, str(curr_idx) + '.pickle'), 'wb') as f:
            pickle.dump(newDict, f)


def main(argv):
    ### initialize
    saveMVFitDir = join(baseDir, 'multiviewFit')
    if not os.path.exists(saveMVFitDir):
        os.mkdir(saveMVFitDir)

    source_handpose = getPredicedPoses()

    print("starting multi-view optimization...")
    numThreads = 1
    numCandidateFrames = len(source_handpose[0])
    numFramesPerThread = np.ceil(numCandidateFrames / numThreads).astype(np.uint32)
    procs = []
    for proc_index in range(numThreads):
        startIdx = proc_index * numFramesPerThread
        endIdx = min(startIdx + numFramesPerThread, numCandidateFrames)
        args = ([], source_handpose[:, startIdx:endIdx], startIdx, saveMVFitDir)
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

    print("end")

    debug_sourceKPS = sourceKPS_np[0]
    debug_fit = fittedKPS_np[0]
    ### visualize ###
    """
    width = 640
    height = 360
    onlyHand_len = len(source_handpose[0])
    for i in range(onlyHand_len):
        data_cam0 = source_handpose[0][i]
        data_cam1 = source_handpose[1][i]
        data_cam2 = source_handpose[2][i]

        img = np.zeros((height, width, 3), dtype=float)

        # data translation
        data_cam0[:, 0] += width / 2.0
        data_cam1[:, 0] += width / 2.0
        data_cam2[:, 0] += width / 2.0

        data_cam0[:, 1] += height / 2.0
        data_cam1[:, 1] += height / 2.0
        data_cam2[:, 1] += height / 2.0

        img_cam0 = manoVis.showHandJoints_cv(img.copy(), np.copy(data_cam0).astype(np.float32))
        img_cam1 = manoVis.showHandJoints_cv(img.copy(), np.copy(data_cam1).astype(np.float32))
        img_cam2 = manoVis.showHandJoints_cv(img.copy(), np.copy(data_cam2).astype(np.float32))
        cv2.imshow("cam0", img_cam0)
        cv2.imshow("cam1", img_cam1)
        cv2.imshow("cam2", img_cam2)
        print("index : ", i)
        cv2.waitKey(0)
    """


if __name__ == '__main__':
    app.run(main)