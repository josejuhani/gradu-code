import numpy as np
import pandas as pd
import os
import random

from BorealWeights import BorealWeightedProblem
from kmeans import kmeans, randomsample
from pyomo.opt import SolverFactory


def nan_to_bau(frame):
    return frame.transpose().fillna(frame.iloc[:, 0]).transpose()


def ideal(dictionary=True):
    if dictionary:
        return {'revenue': 249966739.00009939,
                'deadwood': 218153.21549812937,
                'ha': 20225.257707161425,
                'carbon': 4449001.4721100219}
    else:
        return np.array((249966739.00009939,
                         4449001.4721100219,
                         218153.21549812937,
                         20225.257707161425))


def nadir(dictionary=True):
    if dictionary:
        return {'revenue': 3.09084573e+07,
                'carbon': 2.83139915e+06,
                'deadwood': 8.02116726e+04,
                'ha': 1.19880519e+04}
    else:
        return np.array((3.09084573e+07,
                         2.83139915e+06,
                         8.02116726e+04,
                         1.19880519e+04))


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
    with np.errstate(invalid='ignore'):
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
    return sum([data[ind, int(cluster_list[ind])] * weights[ind]
                for ind in range(len(cluster_list))])


def clusters_to_origin(data, xtoc, cluster_list):
    return sum([sum(data[xtoc == ind][:, int(cluster_list[ind])])
                for ind in range(len(cluster_list))])


def model_to_real_values(data, xtoc, model):
    return clusters_to_origin(data, xtoc, res_to_list(model))


def values_to_list(problem, data):
    lst = []
    for i in problem.model.I:
        for j in problem.model.J:
            if problem.model.x[i, j].value == 1:
                lst.append(data[i, j])
    return lst


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


def calc_ideal_n_nadir(data, xtoc=None, weights=None):
    '''
    Calculate ideal and nadir from the data where different
    objectives are at the "last" level
    '''
    solver = SolverFactory('glpk')
    problems = []
    for i in range(np.shape(data)[-1]):
        problems.append(BorealWeightedProblem(data[:, :, i], weights))

    for j in range(len(problems)):
        solver.solve(problems[j].model)

    if xtoc is None:
        payoff = [[np.sum(values_to_list(problems[j], data[:, :, i]))
                   for i in range(np.shape(data)[-1])]
                  for j in range(len(problems))]
    else:
        payoff = [[np.sum(cluster_to_value(data[:, :, i],
                                           res_to_list(problems[j].model),
                                           weights))
                   for i in range(np.shape(data)[-1])]
                  for j in range(len(problems))]
    ideal = np.max(payoff, axis=0)
    nadir = np.min(payoff, axis=0)
    return ideal, nadir


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
