# copyrights © Mohanad Youssef 2023

import numpy as np

class Sensor(object):
    def __init__(self, model, matrix_R):
        self.model = model
        self.matrix_R = matrix_R
        self.meas = None
        self.time = 0

    def generate(self, actual):
        tf_actual = self.model(actual)
        self.meas = np.random.multivariate_normal(tf_actual, self.matrix_R)
