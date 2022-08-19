import tensorflow as tf
import numpy as np

class Optimizer():
    def __init__(self, loss, varList, type='Adam', learning_rate = 0.01):
        self.loss = loss
        if type == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.999)
        elif type == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimier(learning_rate=1.0)
        elif type == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        elif type == 'Momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum = 0.9)

        self.optOp = self.optimizer.minimize(loss, var_list = varList)
        self.grads = self.optimizer.compute_gradients(loss, varList) #??

        self.numVars = 0 
        for var in varList:
            self.numVars += int(np.prod(var.shape))
        
        if type == 'Adam':
            self.resetInternealTFVars()
        
    def resetInternealTFVars(self):
        varList = []
        vars = self.optimizer.variables()
        for var in vars:
            if 'beta' in var.name:
                varList.append(var)
        self.optIntRestOp = tf.variables_initializer(varList)
        # self.optIntRestOp = tf.variables_initializer(vars)

    def runOptm(self, session, steps, feedDict = None):
        lossCurve = np.zeros((steps,), dtype=np.float32)
        gradCurve = np.zeros((self.numVars, steps), dtype=np.float32)
        for i in range(steps):
            writer = tf.summary.FileWriter('/root/OXR_projects/optimizer/log/', session.graph)
            lt = session.run(self.loss, feed_dict=feedDict)
            writer.close()
            lossCurve[i] = lt
            ind = 0 
            for it in self.grads:
                grad = session.run(it[0], feed_dict = feedDict)
                grad = grad.reshape(-1)
                for j in range(len(grad)):
                    gradCurve[ind, i] = grad[j]
                    ind = ind + 1
            session.run(self.optOp, feed_dict=feedDict)
            if self.type == 'Adam':
                session.run(self.optIntRestOp)
