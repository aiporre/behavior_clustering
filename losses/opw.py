import numpy as np
from scipy.spatial import distance_matrix


class OPWMetric:
    def __init__(self):
        self.p_norm = np.inf
        self.delta = 0.001
        self.lambda_1 = 0.1
        self.lambda_2 = 40.5
        self.maxIter = 1000

    def __call__(self, x, y, *args, **kwargs):
        M, N = x.shape[0], y.shape[0]
        mid = np.sqrt(1 / N ^ 2 + 1 / M ^ 2)
        ii, jj = np.mgrid[1:N + 1, 1:M + 1]
        d = np.abs(ii / N - jj / M) / mid
        P = np.exp(-d ^ 2 / self.delta ^ 2)
        S = self.lambda_1 / ((ii / N - jj / M) ^ 2 + 1)
        a, b = np.ones((N, 1)) / N, np.ones((M, 1)) / M
        D = distance_matrix(x, y, p=2)
        K = P * np.exp((S - D) / self.lambda_2);
        ainvK = np.divide(K, a)
        cnt = 0;
        u = np.ones(N, 1) / N

        # The Sinkhorn's fixed point iteration
        # This part of code is adopted from the code: sinkhornTransport.m by Marco Cuturi
        # website: http://marcocuturi.net/SI.html

        while cnt < self.maxIter:
            u = 1. / (ainvK * np.divide(b, K.T * u));
            cnt += 1;
            # check the stopping criterion every 20 fixed point iterations
            if cnt % 20 == 1 | cnt == self.maxIter:
                # split computations to recover right and left scalings.
                v = np.divide(b, K.T * u)
                u = 1. / (ainvK * v)
                stop_criterion = np.linalg.norm(sum(abs(v * K.T * u - b)), ord=self.p_norm)
                if stop_criterion < self.tolerance | np.isnan(stop_criterion):
                    break
            print('Iteration :', cnt, ' Criterion: ', stop_criterion)


def opw_metric():
    pass
