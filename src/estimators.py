from scipy.special import lambertw
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


class cPCA:
    def __init__(self):
        self.dimension_ = -1
        self.shape = 0

    def mininmum_set_cover(self, data, n_neighbors): # O(n^2*log(n))
        n = data.shape[0]
        temp = NearestNeighbors(n_neighbors=n_neighbors+1)
        temp.fit(data)
        temp = temp.kneighbors_graph(data) # CSR matrix
        q = np.array(temp.sum(axis=0).astype('int32')).squeeze()
        temp = temp.indices.reshape([n, n_neighbors + 1]) # Matrix with indices of neighbors
        taken = np.full(n, True)
        for i in range(n):
            if (q[temp[i]] > 1).all():
                q[temp[i]] -= 1
                taken[i] = False
        return temp[taken]


    def fit(self, data, n_neighbors = 20, alpha = 10, beta = 0.95, P = 0.95, noise = False): # O(n*k + n^2*log(n))
        data = np.array(data)
        n = data.shape[0]
        self.D = distance.cdist(data, data, 'euclidean')

        F = self.mininmum_set_cover(data, n_neighbors)
        dims = []

        for sub in range(F.shape[0]):
            cur_data = np.array(data[F[sub]])
            covariance_matrix = np.cov(cur_data.T)
            eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
            eigen_values = list(reversed(sorted(eigen_values)))
            if noise:
                sum_var = sum(eigen_values)
                cur_sum = 0
                r = 0
                for i in range(data.shape[1] - 1):
                    cur_sum += eigen_values[i]
                    if cur_sum / sum_var <= P and (cur_sum + eigen_values[i + 1]) / sum_var >= P:
                        r = i + 1
                        break
                if r == 0:
                    raise Exception("don't find noise")
                sigma = 0; cnt = 0
                for i in range(max(r, data.shape[1] - 10), data.shape[1]):
                    cnt += 1
                    sigma += eigen_values[i]
                sigma /= cnt
                eigen_values -= sigma
            # criterions
            sum_var = sum(eigen_values)
            cur_sum = 0
            flag = False
            for d in range(len(eigen_values) - 1):
                cur_sum += eigen_values[d]
                if (eigen_values[d] / eigen_values[d + 1] > alpha) or (cur_sum / sum_var > beta):
                    flag = True
                    dims.append(d + 1)
                    break
            if not flag:
                dims.append(len(eigen_values))
        self.dimension_ = np.mean(dims)


class FisherS:
    def __init__(self):
        self.dimension_ = -1


    def fit(self, X, alpha = 0.8, C = 10):
        # centering
        X_mean = np.mean(X, axis=0)
        X -= X_mean
        # apply PCA
        pca = PCA()
        u = pca.fit_transform(X)
        v = pca.components_.T
        s = pca.explained_variance_
        sc = s / s[0]
        ind = np.where(sc > 1 / C)[0]
        X = X @ v[:,ind]
        # whitening
        X = u[:, ind]
        st = np.std(X, axis=0, ddof=1)
        #project on sphere
        st = np.sqrt(np.sum(X**2, axis=1))
        st = np.array([st]).T
        X = X/st
        # Compute the Gram matrix
        xy = X @ X.T
        dxy = np.diag(xy)
        sm = (xy / dxy).T
        sm = sm - np.diag(np.diag(sm))
        sm = sm > alpha
        py = sum(sm.T)
        py = py / len(py)
        separ_fraction = sum(py == 0) / len(py)
        #
        py_mean = np.mean(py)
        n_alpha = np.nan
        if py_mean != 0:
            p = py_mean
            a2 = alpha ** 2
            w = np.log(1 - a2)
            n_alpha = lambertw(-(w/(2*np.pi*p*p*a2*(1-a2))))/(-w)
        if n_alpha == np.inf:
            n_alpha = float('nan')
        self.dimension_ = np.real(n_alpha)






