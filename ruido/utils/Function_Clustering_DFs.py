# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:57:54 2020

MODIFIED VERSION OF LOIC VIENS' clustering code.

The original code can be found at:
https://github.com/lviens/2020_Clustering

Modification:
Return principal component object rather than directly clustering,
with the goal to expand a larger dataset in the principal component
basis, and then return to do the clustering on this larger dataset.

MIT License
Copyright (c) 2020 Loïc Viens

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


"""

import numpy as np
from sklearn.decomposition import PCA 
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator


def run_pca(mat, nr_pc=5):
    """
    Inputs:
        mat: Matrix of correlation functions
        nr_pc: number of principal components to keep or "mle" (will use Bayesian dimensionality selection)
    Outputs:
        pca object (see scikit learn PCA)

    """
    # Perform PCA with the number of principal components given in input

    # cumul_var_perc = 0.
    # n_pca = n_pca_min
    # while cumul_var_perc < min_cumul_var_perc:
    #     pca = PCA(n_components=n_pca)
    #     pca = pca.fit(mat)
    #     # Cumulative variance
    #     cumul_var_perc = np.cumsum(pca.explained_variance_ratio_)[-1]
    #     n_pca += 1
    pca = PCA(n_components=nr_pc)
    pca = pca.fit(mat)
    cumul_var_perc = np.cumsum(pca.explained_variance_ratio_)[-1]

    print('The first ' + str(nr_pc) + ' PCs explain ' + str(cumul_var_perc) +
          ' % of the cumulative variance')

    return(pca)


def gmm(matpc, range_GMM=None, fixed_nc=None, max_iter=10000,
        n_init=1, tol=1.e-3, reg_covar=1.e-6, verbose=False):


    if range_GMM is not None:
        # Compute the GMM for different number of clusters set by "range_GMM"
        models = [GaussianMixture(n, covariance_type='full', random_state=0,
                                  max_iter=max_iter, tol=tol,
                                  reg_covar=reg_covar).fit(matpc) for n in range_GMM]
        # Compute the Bayesian information criterion (BIC) for each model
        BICF = [m.bic(matpc) / 1000 for m in models]
        # Determine the best number of clusters using the knee (or elbow)
        kn = KneeLocator(range_GMM, BICF, S=1, curve='convex',
                         direction='decreasing')
        n_clusters = kn.knee
        if n_clusters is None:
            n_clusters = min(range_GMM)
    else:
        models = []
        n_clusters = fixed_nc
        BICF = None

    # Perform clustering for the best number of clusters
    if n_init > 1:
        init_type = "random"
    else:
        init_type = "kmeans"
    gmix = GaussianMixture(n_components=n_clusters, covariance_type='full',
                           max_iter=max_iter, tol=tol, n_init=n_init,
                           reg_covar=reg_covar, verbose=verbose,
                           verbose_interval=max_iter//10,
                           init_params=init_type)
    gmix.fit(matpc)
    gmixfinPCA = gmix.predict(matpc)
    probs = gmix.predict_proba(matpc)

    return models, n_clusters, gmixfinPCA, probs, BICF
