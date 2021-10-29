import numpy as np
from sklearn.decomposition import PCA
import sys


# returns choosing how many main factors
def index_lst(lst, component=0, rate=0):
    # component: numbers of main factors
    # rate: rate of sum(main factors)/sum(all factors)
    # rate range suggest: (0.8,1)
    # if you choose rate parameter, return index = 0 or less than len(lst)
    if component and rate:
        print('Component and rate must choose only one!')
        sys.exit(0)
    if not component and not rate:
        print('Invalid parameter for numbers of components!')
        sys.exit(0)
    elif component:
        print('Choosing by component, components are %s......' % component)
        return component
    else:
        print('Choosing by rate, rate is %s ......' % rate)
        for i in range(1, len(lst)):
            if sum(lst[:i]) / sum(lst) >= rate:
                return i
        return 0


def main():
    # test data
    mat = [[-1, -1, 0, 2, 1], [2, 0, 0, -1, -1], [2, 0, 1, 1, 0]]

    # simple transform of test data
    Mat = np.array(mat, dtype='float64')
    print('Before PCA transforMation, data is:\n', Mat)
    print('\nMethod 1: PCA by original algorithm:')
    p, n = np.shape(Mat)  # shape of Mat
    t = np.mean(Mat, 0)  # mean of each column

    # substract the mean of each column
    for i in range(p):
        for j in range(n):
            Mat[i, j] = float(Mat[i, j] - t[j])

    # covariance Matrix
    print(Mat.shape)
    cov_Mat = np.dot(Mat.T, Mat) / (p - 1)
    print(cov_Mat.shape)

    # PCA by original algorithm
    # eigvalues and eigenvectors of covariance Matrix with eigvalues descending
    U, V = np.linalg.eigh(cov_Mat)
    # Rearrange the eigenvectors and eigenvalues
    U = U[::-1]
    for i in range(n):
        V[i, :] = V[i, :][::-1]
    # choose eigenvalue by component or rate, not both of them euqal to 0
    Index = index_lst(U, component=2)  # choose how many main factors
    if Index:
        v = V[:, :Index]  # subset of Unitary matrix
    else:  # improper rate choice may return Index=0
        print('Invalid rate choice.\nPlease adjust the rate.')
        print('Rate distribute follows:')
        print([sum(U[:i]) / sum(U) for i in range(1, len(U) + 1)])
        sys.exit(0)
    # data transformation
    T1 = np.dot(Mat, v)
    # print the transformed data
    print('We choose %d main factors.' % Index)
    print('After PCA transformation, data becomes:\n', T1)

    # PCA by original algorithm using SVD
    print('\nMethod 2: PCA by original algorithm using SVD:')
    # u: Unitary matrix, eigenvectors in columns
    # d: list of the singular values, sorted in descending order
    u, d, v = np.linalg.svd(cov_Mat)
    Index = index_lst(d, rate=0.95)  # choose how many main factors
    T2 = np.dot(Mat, u[:, :Index])  # transformed data
    print(cov_Mat.shape)
    print(Mat.shape)
    print(u.shape)
    print('We choose %d main factors.' % Index)
    print('After PCA transformation, data becomes:\n', T2)

    # PCA by Scikit-learn
    pca = PCA(n_components=2)  # n_components can be integer or float in (0,1)
    pca.fit(mat)  # fit the model
    print('\nMethod 3: PCA by Scikit-learn:')
    print('After PCA transformation, data becomes:')
    print(pca.fit_transform(mat))  # transformed data


if __name__ == '__main__':
    mat = np.array([[-55, -44, 0, 23, 76], [-12, -20, 10, 11, 11]])
    print(mat)

    C = mat @ mat.T / 5
    # C = np.cov(mat)
    print(C)

    eig_val, eig_vec = np.linalg.eig(C)
    print(eig_val)
    print(eig_vec.T)

    P = eig_vec.T[:1, :]
    print(P)

    res = P @ mat
    print(res)

    # 用sklearn的PCA
    pca = PCA(n_components=1)
    pca.fit(mat.T)
    print(pca.transform(mat.T).T)
