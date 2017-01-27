import numpy as np
import scipy.sparse as sps
from .base import check_matrix
from .._cython._similarity import memory_cosine_common, cosine_common, pearson_corr


class ISimilarity(object):
    """Abstract interface for the similarity metrics"""

    def __init__(self, shrinkage=10):
        self.shrinkage = shrinkage

    def compute(self, X):
        pass


class Cosine(ISimilarity):
    def compute(self, X):
        # convert to csc matrix for faster column-wise operations
        X = check_matrix(X, 'csc', dtype=np.float32)

        # 1) normalize the columns in X
        # compute the column-wise norm
        # NOTE: this is slightly inefficient. We must copy X to compute the column norms.
        # A faster solution is to  normalize the matrix inplace with a Cython function.
        Xsq = X.copy()
        Xsq.data **= 2
        norm = np.sqrt(Xsq.sum(axis=0))
        norm = np.asarray(norm).ravel()
        norm += 1e-6
        # compute the number of non-zeros in each column
        # NOTE: this works only if X is instance of sparse.csc_matrix
        col_nnz = np.diff(X.indptr)
        # then normalize the values in each column
        X.data /= np.repeat(norm, col_nnz)

        # 2) compute the cosine similarity using the dot-product
        dist = X.T.dot(X).toarray()
        # zero out diagonal values
        np.fill_diagonal(dist, 0.0)
        if self.shrinkage > 0:
            dist = self.apply_shrinkage(X, dist)
        return dist

    def apply_shrinkage(self, X, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind.T.dot(X_ind).toarray().astype(np.float32)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        dist *= co_counts / (co_counts + self.shrinkage)
        return dist


class Pearson(ISimilarity):
    def compute(self, X):
        # convert to csc matrix for faster column-wise operations
        X = check_matrix(X, 'csc', dtype=np.float32)
        # subtract the item average rating
        col_nnz = np.diff(X.indptr)
        col_means = np.asarray(X.sum(axis=0) / (col_nnz + 1e-6)).ravel()
        X.data -= np.repeat(col_means, col_nnz)

        dist, co_counts = cosine_common(X)
        if self.shrinkage > 0:
            dist *= co_counts / (co_counts + self.shrinkage)
        return dist

    def memory_compute(self, X, k_top_value, nitems):
        # convert to csc matrix for faster column-wise operations
        X = check_matrix(X, 'csc', dtype=np.float32)
        # subtract the item average rating
        col_nnz = np.diff(X.indptr)
        col_means = np.asarray(X.sum(axis=0) / (col_nnz + 1e-6)).ravel()
        X.data -= np.repeat(col_means, col_nnz)

        # for dist, co_counts in memory_cosine_common:
        #     if self.shrinkage > 0:
        #         dist *= co_counts / (co_counts + self.shrinkage)
        #
        #     yield dist

        # # Delivering only patches of the data.
        # n_items = dist.shape[1]
        # linspace = np.linspace(0, n_items, num=29, endpoint=False)
        #
        # lower_bound = 0
        # for upper_bound in linspace[1:]:
        #     yield dist[:, lower_bound:int(upper_bound)]
        #     lower_bound = int(upper_bound)

        # dist, co_counts, indices_row, indices_col = memory_cosine_common(X)
        # if self.shrinkage > 0:
        #     dist *= co_counts / (co_counts + self.shrinkage)

        # return dist

        return memory_cosine_common(X, self.shrinkage, k_top_value, nitems)

    def pearson(self, X):
        # It' assumed that every col has already the item mean substracted.
        ncols = X.shape[1]
        sim = np.empty(shape=(ncols, ncols), dtype=float)

        # is the standard CSC representation where the row indices for column i are stored in
        # indices[indptr[i]:indptr[i+1]]
        # and their corresponding values are stored in
        # data[indptr[i]:indptr[i+1]].
        # If the shape parameter is not supplied, the matrix dimensions are inferred from the index arrays.
        data = X.data
        indptr = X.indptr
        indices = X.indices



        # first col.
        for i in range(ncols):
            # To get the number of non-empty rows of col i.
            max_row_i = indptr[i + 1] - indptr[i]

            # second col.
            for j in range(i+1, ncols):
                # To get the number of non-empty rows of col j.
                max_row_j = indptr[j + 1] - indptr[j]

                # Checking the common rows, if any.
                curr_row_i = 0
                curr_row_j = 0
                n_common_cols = 0
                ij_sum, ii_sum, jj_sum = 0.0, 0.0, 0.0
                while curr_row_i < max_row_i and curr_row_j < max_row_j:
                    if indices[indptr[i] + curr_row_i] < indices[indptr[j] + curr_row_j]:
                        curr_row_i += 1 # We need to approach the current row in i to see if they're the same.

                    elif indices[indptr[i] + curr_row_i] > indices[indptr[j] + curr_row_j]:
                        curr_row_j += 1 # We need to approach the current row in j to see if they're the same.

                    else: # Here we have a common row for both cols.
                        x_i = data[indptr[i] + curr_row_i]
                        x_j = data[indptr[j] + curr_row_j]
                        ij_sum += x_i * x_j # Numerator mult.
                        ii_sum += x_i ** 2 # i sum in the denominator.
                        jj_sum += x_j ** 2 # j sum in the denominator.
                        curr_row_i += 1
                        curr_row_j += 1
                        n_common_cols += 1

                # If we found common rows in the cols.
                if n_common_cols > 0:
                    sim[i, j] = ij_sum / np.sqrt(ii_sum) * np.sqrt(jj_sum)
                    sim[j, i] = sim[i, j]
                else:
                    sim[i, j] = 0.0
                    sim[j, i] = 0.0

            # Set the diagonals to zero
            np.fill_diagonal(sim, 0.0)

        return sim, 1


class AdjustedCosine(ISimilarity):
    def compute(self, X):
        # convert X to csr matrix for faster row-wise operations
        X = check_matrix(X, 'csr', dtype=np.float32)
        # subtract the user average rating
        row_nnz = np.diff(X.indptr)
        row_means = np.asarray(X.sum(axis=1).ravel() / (row_nnz + 1e-6)).ravel()
        X.data -= np.repeat(row_means, row_nnz)

        # convert X to csc before applying cosine_common
        X = X.tocsc()
        dist, co_counts = cosine_common(X)
        if self.shrinkage > 0:
            dist *= co_counts / (co_counts + self.shrinkage)
        return dist


# from .._cython._similarity import pearson_corr
# class Pearson2(IDistance):
#     def compute(self, X):
#         # convert to csc matrix for faster column-wise operations
#         X = check_matrix(X, 'csc', dtype=np.float32)
#         dist, co_counts = pearson_corr(X)
#         if self.shrinkage > 0:
#             dist *= co_counts / (co_counts + self.shrinkage)
#         return dist

