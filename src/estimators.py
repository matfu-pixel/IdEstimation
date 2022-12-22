import skdim
from scipy.special import lambertw
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance

class cPCA:
    def __init__(self):
        self.dimension_ = -1
        self.shape = 0

    def mininmum_set_cover(self, data, n_neighbors): # O(n^2*log(n))
        n = data.shape[0]
        Q = np.zeros(n)
        F = [[] for _ in range(n)]
        g = np.zeros((n, n))
        for i in range(n):
            tmp = []
            for j in range(n):
                tmp.append((self.D[i][j], j))
            tmp.sort()
            for j in range(min(n_neighbors, len(tmp))):
                F[i].append(tmp[j][1])
                g[i][tmp[j][1]] = 1

        for i in range(n):
            for j in range(n):
                Q[i] += g[j][i]

        ans = []
        for i in range(n):
            flag = True
            for j in F[i]:
                if Q[j] <= 1:
                    flag = False
                    break
            if flag:
                for j in F[i]:
                    Q[j] -= 1
            else:
                ans.append(i)
        final = [[] for _ in range(len(ans))]
        cnt = 0
        for i in ans:
            for j in F[i]:
                final[cnt].append(j)
            cnt += 1
        return final
        

    def fit(self, data, n_neighbors = 20, alpha = 10, beta = 0.95, P = 0.95, noise = False): # O(n*k + n^2*log(n))
        print(1, end="")
        data = np.array(data)
        n = data.shape[0]
        self.D = distance.cdist(data, data, 'euclidean')
        F = self.mininmum_set_cover(data, n_neighbors)
        dims = []

        for ids in F:
            cur_data = np.zeros((len(ids), data.shape[1])) 
            cur_data = np.array(data[ids])
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

        




