import numpy as np
# from scipy import stats
# from numpy import matlib as mb
from sklearn import linear_model
import statsmodels.api as sm

def nn(data, train_len, embed_dim, k_nearest, method='correlation', n_out_sample=0):
    """ nearest neighbour forecast method"""
    # train_len - the observation where the InSample forecasts will start. It also defines the training period of the
    # algorithm. For example, if len(data)=500 and train_len=400, the values of 1:400 will be the training period for
    # the forecasted value of 401. For the forecast of 402, the training period is 1:401, meaning that each time a new
    # observation is available, the algorithm adds it to the training period. Please notes that the parameter train_len
    # doesn't have any effect on the out of sample forecasts.

    # embed_dim - Embedding dimension (size of the histories)"

    # k_nearest - The number of nearest neighbours to be used in the construction of the forecasts

    in_sample_fore = []
    for v in range(len(data) - train_len):
        curr_train = data[:train_len + v]
        curr_fore = nn_core(curr_train, embed_dim, k_nearest, method)
        in_sample_fore.append(curr_fore)
    in_sample_res = data[train_len:] - in_sample_fore
    out_sample_fore = []
    if n_out_sample != 0:
        curr_train = data
        for z in range(1, n_out_sample):
            out_sample_tmp = nn_core(curr_train, embed_dim, k_nearest, method)
            out_sample_fore.append(out_sample_tmp)
            curr_train.append(out_sample_tmp)
    return out_sample_fore, in_sample_fore, in_sample_res

def nn_core(data, embed_dim, k_nearest, method='correlation'):
    """ core method for nearest_neighbour"""
    n1 = len(data)
    mcorrel = [None] * (n1-embed_dim)
    if method == "correlation":
        chunk = np.ndarray(embed_dim)
        chunk[:] = data[n1 - embed_dim: n1]
        for i in range(n1 - embed_dim):
            aba = np.corrcoef(chunk, data[n1 - embed_dim - i - 1: n1 - i - 1])
            mcorrel[n1-embed_dim-i-1] = aba[1][0]
        mcorrel2 = np.abs(mcorrel)
        idx = np.argsort(mcorrel2)
        full_idx = np.transpose(np.tile(idx[len(idx)-k_nearest:], (embed_dim + 1, 1))) + np.tile(range(embed_dim + 1), (k_nearest, 1))
        s = np.ndarray(full_idx.shape)
        for i in range(full_idx.shape[0]):
            for j in range(full_idx.shape[1]):
                s[i][j] = data[full_idx[i][j]]
        X = s[:, range(embed_dim-1, -1, -1)]
        Y = s[:, embed_dim]

        # regr = linear_model.LinearRegression()
        # regr.fit(X, Y)
        # intercept = regr.intercept_
        # slope = regr.coef_
        # coef = np.empty(embed_dim+1)
        # coef[0] = intercept
        # coef[1:] = slope

        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.concatenate((np.ones((k_nearest, 1)), s[:, range(embed_dim, 0, -1)]), axis=1), s[:, embed_dim])

        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        coef = model._results.params

        c = np.ones(embed_dim+1)
        c[1:] = data[range(n1-1, n1-embed_dim-1, -1)]

        result = np.matmul(c, coef)

        return result

    elif method == "absolute_distance":
        chunk = np.ndarray(embed_dim)
        chunk[:] = data[n1 - embed_dim: n1]
        sum_distance = [None] * (n1 - embed_dim)
        for i in range(n1 - embed_dim):
            distance = np.sum(np.abs(chunk-data[n1 - embed_dim - i - 1: n1 - i - 1]))
            sum_distance[n1-embed_dim-i-1] = distance
        idx = np.argsort(sum_distance)[::-1]
        full_idx = np.transpose(np.tile(idx[len(idx)-k_nearest:], (embed_dim + 1, 1))) + np.tile(range(embed_dim + 1), (k_nearest, 1))
        s = np.ndarray(full_idx.shape)
        for i in range(full_idx.shape[0]):
            for j in range(full_idx.shape[1]):
                s[i][j] = data[full_idx[i][j]]
        result = np.mean(s[:, embed_dim])
        #print(result)

        return result



