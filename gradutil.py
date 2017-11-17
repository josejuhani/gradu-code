import numpy as np
import pandas as pd
import os
import random

from BorealWeights import BorealWeightedProblem
from kmeans import kmeans, randomsample
from pyomo.opt import SolverFactory


def real_solutions():
    return {'revenue': 249966739.00009939,
            'deadwood': 218153.21549812937,
            'ha': 10327.079086726841,
            'carbon': 4449001.4721100219}


def init_boreal():
    data_dir = os.path.join(os.getcwd(), '../boreal_data')
    carbon = pd.read_csv(os.path.join(data_dir, 'Carbon_storage.csv'))
    HA = pd.read_csv(os.path.join(data_dir, 'Combined_HA.csv'))
    deadwood = pd.read_csv(os.path.join(data_dir, 'Deadwood_volume.csv'))
    revenue = pd.read_csv(os.path.join(data_dir, 'Timber_revenues.csv'))
    return revenue, carbon, deadwood, HA


def normalize(data):
    norm_data = data.copy()
    inds = np.where(np.isnan(norm_data))
    norm_data[inds] = np.take(np.nanmin(norm_data, axis=0)
                              - np.nanmax(norm_data, axis=0),
                              inds[1])
    norm_data -= np.min(norm_data, axis=0)
    with np.errstate(divide='ignore'):
        normax = np.max(norm_data, axis=0)
        norm_data = np.where(normax != 0., norm_data / normax, 0)
    return norm_data


def optimize_all(combined_data, weights, opt):
    problem1 = BorealWeightedProblem(combined_data[:, :7], weights)
    res1 = opt.solve(problem1.model, False)

    problem2 = BorealWeightedProblem(combined_data[:, 7:14], weights)
    res2 = opt.solve(problem2.model, False)

    problem3 = BorealWeightedProblem(combined_data[:, 14:21], weights)
    res3 = opt.solve(problem3.model, False)

    problem4 = BorealWeightedProblem(combined_data[:, 21:], weights)
    res4 = opt.solve(problem4.model, False)

    return (problem1, res1), (problem2, res2),\
           (problem3, res3), (problem4, res4)


def res_to_list(model):
    resdict = model.x.get_values()
    reslist = np.zeros(model.n.value)
    for i, j in resdict.keys():
        if resdict[i, j] == 1.:
            reslist[i] = j
    return reslist


def cluster_to_value(data, cluster_list, weights):
    return sum([
        data[ind, int(cluster_list[ind])] * weights[ind]
        for ind in range(len(cluster_list))])


def clusters_to_origin(data, xtoc, cluster_list):
    return sum([sum(data[xtoc == ind][:, int(cluster_list[ind])])
                for ind in range(len(cluster_list))])


def model_to_real_values(data, xtoc, model):
    return clusters_to_origin(data, xtoc, res_to_list(model))


def cluster(data, nclust, seed, delta=.0001, maxiter=100,
            metric='cosine', verbose=1):
    random.seed(seed)
    np.random.seed(seed)
    data[np.isnan(data)] = np.nanmin(data) - np.nanmax(data)
    randomcenters = randomsample(data, nclust)
    centers, xtoc, dist = kmeans(data,
                                 randomcenters,
                                 delta=delta,
                                 maxiter=maxiter,
                                 metric=metric,
                                 verbose=verbose)
    return centers, xtoc, dist


def cNopt(orig_data, clust_data, opt_data, opt, nclust='10', seed=2):
    c, xtoc, dist = cluster(clust_data, nclust, seed, verbose=0)
    weights = np.array([sum(xtoc == i) for i in range(len(c))])
    opt_x = np.array([opt_data[xtoc == i].mean(axis=0)
                      for i in range(nclust)])
    problem1, problem2, problem3, problem4 = optimize_all(opt_x,
                                                          weights,
                                                          opt)
    res1 = model_to_real_values(orig_data[:, :7], xtoc, problem1[0].model)
    res2 = model_to_real_values(orig_data[:, 7:14], xtoc, problem2[0].model)
    res3 = model_to_real_values(orig_data[:, 14:21], xtoc, problem3[0].model)
    res4 = model_to_real_values(orig_data[:, 21:], xtoc, problem4[0].model)

    return res1, res2, res3, res4


if __name__ == '__main__':
    seed = 2
    nclust = 50
    revenue, carbon, deadwood, ha = init_boreal()
    x = np.concatenate((revenue, carbon, deadwood, ha), axis=1)

    norm_x = np.concatenate((normalize(revenue.values),
                             normalize(carbon.values),
                             normalize(deadwood.values),
                             normalize(ha.values)), axis=1)

    no_nan_x = x.copy()
    inds = np.where(np.isnan(no_nan_x))
    no_nan_x[inds] = np.take(np.nanmin(no_nan_x, axis=0)
                             - np.nanmax(no_nan_x, axis=0), inds[1])
    opt = SolverFactory('glpk')

    res1, res2, res3, res4 = cNopt(x, norm_x, no_nan_x, opt, nclust, seed)

    print('Results when surrogate mapped to real values:')
    print('(i) Harvest revenues {: .0f} M€'.format(res1/1000000))
    print('(ii) Carbon storage {: .0f} x 100 MgC'.format(res2/100))
    print('(iii) Deadwood index {: .0f} m3'.format(res3))
    print('(iv) Combined Habitat {: .0f}'.format(res4))
