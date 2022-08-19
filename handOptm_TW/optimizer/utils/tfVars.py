import tensorflow as tf
import json
import numpy as np
from os.path import join

baseDir = './test_tw'
handposeRoot = join(baseDir, 'onlyHandWorldCoordinate_uvd.json')

def varInit(ref):
    with open(handposeRoot, 'r') as fi:
        data = json.load(fi)
    handpose_0 = np.array(data['0_%d'%ref])
    handpose_1 = np.array(data['1_%d'%ref])
    handpose_2 = np.array(data['2_%d'%ref]) #[116, 21, 2]
    initVars = []
    with tf.variable_scope('hand', reuse=tf.AUTO_REUSE):
        for idx in range(handpose_0.shape[0]):
            init_joint = np.mean([handpose_0[idx], handpose_1[idx], handpose_2[idx]], axis=0)
            jointVar = tf.get_variable(name='%d_joint'%idx, initializer = init_joint, dtype=tf.float64)
            initVars.append(jointVar) #len 116

    refpose_0 = np.array(data['0_0'])
    refpose_1 = np.array(data['1_1'])
    refpose_2 = np.array(data['2_2'])
    refpose_set = np.array([refpose_0, refpose_1, refpose_2])
    return tf.Variable(initVars), refpose_set
    

if __name__ == '__main__':
    a, b = varInit(2)
    print(a.shape)
