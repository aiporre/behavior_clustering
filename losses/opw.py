import numpy as np
from  scipy.spatial import distance_matrix
class OPWMetric:
    def __init__(self):
        self.p_norm = np.inf
        self.delta = 0.001
        self.lambda_1 = 0.1
        self.lambda_2 = 40.5
    def __call__(self, x, y, *args, **kwargs):
        M, N = x.shape[0], y.shape[0]
        P = np.zeros(M,N)
        S = np.zeros(M, N)
        mid = np.sqrt(1/N^2+1/M^2)
        ii, jj = np.mgrid[1:N+1,1:M+1]
        d = np.abs(ii/N-jj/M)/mid
        P = np.exp(-d^2/self.delta^2)
        S = self.lambda_1/((ii/N-jj/M)^2+1)
        a, b = np.ones((N,1))/N, np.ones((M,1))/M
        D = distance_matrix(x, y, p=2)
        K = P * np.exp((S - D)/ self.lamda_2);
        ainvK = K/a
        compt = 0;
        u = np.ones(N, 1) / N;

def opw_metric():
    pass