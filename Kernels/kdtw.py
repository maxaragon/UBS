from scipy.spatial.distance import cdist
import numpy as np

def kdtw_lk(A, B, local_kernel):
    d=np.shape(A)[1]
    Z=[np.zeros(d)]
    A = np.concatenate((Z,A), axis=0)
    B = np.concatenate((Z,B), axis=0)
    [la,d] = np.shape(A)
    [lb,d] = np.shape(B)
    DP = np.zeros((la,lb))
    DP1 = np.zeros((la,lb));
    DP2 = np.zeros(max(la,lb));
    l=min(la,lb);
    DP2[1]=1.0;
    for i in range(1,l):
        DP2[i] = local_kernel[i-1,i-1];

    DP[0,0] = 1;
    DP1[0,0] = 1;
    n = len(A);
    m = len(B);

    for i in range(1,n):
        DP[i,1] = DP[i-1,1]*local_kernel[i-1,2];
        DP1[i,1] = DP1[i-1,1]*DP2[i];

    for j in range(1,m):
        DP[1,j] = DP[1,j-1]*local_kernel[2,j-1];
        DP1[1,j] = DP1[1,j-1]*DP2[j];

    for i in range(1,n):
        for j in range(1,m): 
            lcost=local_kernel[i-1,j-1];
            DP[i,j] = (DP[i-1,j] + DP[i,j-1] + DP[i-1,j-1]) * lcost;
            if i == j:
                DP1[i,j] = DP1[i-1,j-1] * lcost + DP1[i-1,j] * DP2[i] + DP1[i,j-1]  *DP2[j]
            else:
                DP1[i,j] = DP1[i-1,j] * DP2[i] + DP1[i,j-1] * DP2[j];
    DP = DP + DP1;
    return DP[n-1,m-1]

def kdtw(A, B, sigma = 1.0, epsilon = 1e-3):
    distance = cdist(A, B, 'sqeuclidean')
    local_kernel = (np.exp(-distance/sigma)+epsilon)/(3*(1+epsilon))
    return kdtw_lk(A,B,local_kernel)