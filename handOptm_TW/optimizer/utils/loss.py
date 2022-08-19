import tensorflow as tf
import numpy as np
import json
from utils.uvd_transform import uvdtouvd

Ks = np.load('/root/OXR_projects/optimizer/intrinsic.npy') #0,1,2
with open('/root/OXR_projects/optimizer/extrinsic.json', 'r') as f:
    ext = json.load(f) #0_ref, 1_ref, 2_ref
NUM_VIEWS = 3

class LossFunc():
    def __init__(self, initVars, refpose, refcam):
        self.vars = initVars
        self.refpose = refpose
        self.refcam = refcam
        self.K_ref = Ks[refcam]
        self.ext_ref = ext['ref%d'%refcam]
    def getlossFunc(self):
        loss = tf.reduce_mean(uvdtouvd(self.vars, self.K_ref, Ks[0], self.ext_ref[0])[:,:,:-1] - self.refpose[0][:,:,:-1]
                                                +uvdtouvd(self.vars, self.K_ref, Ks[1], self.ext_ref[1])[:,:,:-1] - self.refpose[1][:,:,:-1]
                                                +uvdtouvd(self.vars, self.K_ref, Ks[2], self.ext_ref[2])[:,:,:-1] - self.refpose[2][:,:,:-1])
        return loss

if __name__ == '__main__':
    print(Ks.shape)
    print(np.array(ext['ref2']).shape)

        
