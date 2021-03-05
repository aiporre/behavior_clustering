import numpy as np
from scipy.spatial import distance_matrix


class OPWMetric:
    def __init__(self):
        self.p_norm = np.inf
        self.delta = 1.0
        self.lambda_1 = 50
        self.lambda_2 = 0.1
        self.maxIter = 20
        self.tolerance = 0.5e-2

    def __call__(self, x, y, *args, **kwargs):
        N, M = x.shape[0], y.shape[0]
        mid = np.sqrt(1 / N ** 2 + 1 / M ** 2)
        ii, jj = np.mgrid[1:N + 1, 1:M + 1]
        d = np.abs(ii / N - jj / M) / mid
        P = np.exp(-d ** 2 / (2 * self.delta ** 2))/ (self.delta*np.sqrt(2*np.pi))
        S = self.lambda_1 / ((ii / N - jj / M) ** 2 + 1)
        a, b = np.ones((N, 1)) / N, np.ones((M, 1)) / M
        D = distance_matrix(x, y, p=2)
        K = P * np.exp((S - D) / self.lambda_2)
        K_tilde = K / a  # diag(1/a)*K
        cnt = 0
        u = np.ones((N, 1)) / N

        # The Sinkhorn's fixed point iteration
        # This part of code is adopted from the code: sinkhornTransport.m by Marco Cuturi
        # website: http://marcocuturi.net/SI.html

        while cnt < self.maxIter:
            u = 1. / np.dot(K_tilde, (b / np.dot(K.T, u)))
            # print('u = ', u)
            cnt += 1
            # check the stopping criterion every 20 fixed point iterations
            if cnt % 20 == 1 or cnt == self.maxIter:
                # split computations to recover right and left scaling.
                v = b / np.dot(K.T, u)
                u = 1. / np.dot(K_tilde, v)
                stop_criterion = np.linalg.norm(sum(abs(v * np.dot(K.T, u) - b)), ord=self.p_norm)
                print(' ==> stop criterion: ', stop_criterion)
                if stop_criterion < self.tolerance or np.isnan(stop_criterion):
                    break
                print('Iteration :', cnt, ' Criterion: ', stop_criterion)
                cnt += 1

        U = K * D
        # v = b / np.dot(K.T, u)
        distance = sum(u * np.dot(U, v))
        transport = u * K * v.T
        return distance, transport
    def calculate_assigment(self, x, y, only_indices=False):
        d, T = self(x,y)
        if only_indices:
            x_assigment = np.argmax(T, axis=1)
            y_assigment = np.argmax(T, axis=0)
        else:
            x_assigment = y[np.argmax(T, axis=1)]
            y_assigment = x[np.argmax(T, axis=0)]

        return x_assigment, y_assigment


def opw_metric():
    pass
