from pyomo.opt import SolverFactory
from BorealWeights import BorealWeightedProblem
import numpy as np
import pandas as pd
import gradutil as gu
import simplejson as json
from scipy.spatial.distance import euclidean
from time import time


def clustering_to_optims(x_orig, x_clust, x_opt, names, clustering, opt,
                         logger=None, starttime=None):
    from time import time
    optims = dict()
    for nclust in sorted(clustering.keys()):
        n_optims = dict()
        for seedn in clustering[nclust].keys():
            xtoc = np.array(clustering[nclust][seedn]['xtoc'])
            w = np.array([sum(xtoc == i)
                          for i in range(nclust)
                          if sum(xtoc == i) > 0])
            # Calculate the euclidian center of the cluster (mean)
            # and then the point closest to that center according to
            # euclidian distance, and then use the data format meant
            # for optimization
            c_close = np.array([
                x_opt[min(np.array(range(len(xtoc)))[xtoc == i],
                          key=lambda index:
                          euclidean(x_clust[index],
                                    np.mean(x_clust[xtoc == i],
                                            axis=0)))]
                for i in range(nclust) if sum(xtoc == i) > 0
            ])
            problems = [BorealWeightedProblem(c_close[:, :, i], weights=w)
                        for i in range(np.shape(c_close)[-1])]
            for p in problems:
                opt.solve(p.model)
            n_optims[seedn] = dict()
            for ind, name in enumerate(names):
                n_optims[seedn][name] = dict()
                n_optims[seedn][name]['real'] = gu.model_to_real_values(
                    x_orig[:, :, ind],
                    problems[ind].model,
                    xtoc)
                n_optims[seedn][name]['surrogate'] = gu.cluster_to_value(
                    x_orig[:, :, ind], gu.res_to_list(problems[ind].model), w)
            if logger:
                logger.info('Optimized {} clusters. Seed {}'.format(nclust,
                                                                    seedn))
            if starttime:
                logger.info('Since start {:2.0f} s.'.format(time()-starttime))
        optims[nclust] = n_optims
        if logger:
            logger.info('Optimized {} clusters with every seed'.format(nclust))
        if starttime:
            logger.info('Since start {:2.0f}s.'.format(time()-starttime))
        with open('optimizations/dump_all{}.json'.format(nclust), 'w') as file:
            json.dump(n_optims, file)
    return optims


def clustering_to_dict(readfile):
    with open(readfile, 'r') as rfile:
        clustering = json.loads(rfile.read())

    new_clustering = dict()
    for nclust in clustering.keys():
        new_clustering[eval(nclust)] = dict()
        for seedn in clustering[nclust].keys():
            new_clustering[eval(nclust)][eval(seedn)] = dict()
            for key in clustering[nclust][seedn].keys():
                new_clustering[eval(nclust)][eval(seedn)][key] = \
                                    np.array(clustering[nclust][seedn][key])
    return new_clustering


def eval_dists(clustering, key):
    distsum = dict()
    for nclust in clustering[key]:
        dists = dict()
        for seedn in nclust.keys():
            clust = nclust[seedn]
            dists[seedn] = np.nansum(clust['dist'])
        distsum[nclust] = dists
    return distsum


def clustering(x, nclusts, seeds, logger=None, starttime=None):
    res = dict()
    for nclust in nclusts:
        res_clust = dict()
        for seedn in seeds:
            c, xtoc, dist = gu.cluster(x, nclust, seedn, verbose=0)
            res_clust[seedn] = {'c': c.tolist(),
                                'xtoc': xtoc.tolist(),
                                'dist': dist.tolist()}
            if logger:
                logger.info('Clustered to {} clusters. Seed {}'.format(nclust,
                                                                       seedn))
            if starttime:
                logger.info('Since start {:2.0f}s.'.format(time()-starttime))
        res[nclust] = res_clust
        if logger:
            logger.info('Clustered to {:2.0f} clusters'.format(nclust))
        if starttime:
            logger.info('Since start {:2.0f}s.'.format(time()-starttime))
        with open('clusterings/dump{}.json'.format(nclust), 'w') as file:
            json.dump(res_clust, file)
    return res


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    start = time()
    logger = logging.getLogger(__name__)
    logger.info('Started')
    revenue, carbon, deadwood, ha = gu.init_boreal()
    n_revenue = gu.nan_to_bau(revenue)
    n_carbon = gu.nan_to_bau(carbon)
    n_deadwood = gu.nan_to_bau(deadwood)
    n_ha = gu.nan_to_bau(ha)
    revenue_norm = gu.normalize(n_revenue.values)
    carbon_norm = gu.normalize(n_carbon.values)
    deadwood_norm = gu.normalize(n_deadwood.values)
    ha_norm = gu.normalize(n_ha.values)
    ide = gu.ideal(False)
    nad = gu.nadir(False)
    opt = SolverFactory('cplex')
    x = pd.concat((n_revenue, n_carbon, n_deadwood, n_ha), axis=1)
    x_stack = np.dstack((n_revenue, n_carbon, n_deadwood, n_ha))
    x_norm = gu.normalize(x.values)
    x_norm_stack = gu.normalize(x_stack)
    nclusts1 = range(100, 8500, 200)
    seeds = range(2, 12)
    logger.info('Initialized. Since start {:2.0f} sec'.format(time()-start))
    '''
    clustering1 = clustering(x_norm, nclusts1, seeds, logger, start)
    with open('clusterings/dump8500all.json', 'w') as file:
        json.dump(clustering1, file)
    logger.info('All clustered to 8500. Time since start {}.'.
                format(time()-start))

    nclusts2 = range(50, 600, 100)
    logger.info('Doing some extra clusterings. {}'.format(time()-start))
    clustering2 = clustering(x_norm, nclusts2, seeds, logger, start)
    with open('clusterings/dump600all.json', 'w') as file:
        json.dump(clustering2, file)
    logger.info('All clustered too 600 (first round). Time since start {}.'.
                format(time()-start))

    nclusts3 = range(200, 601, 200)
    logger.info('Still some more extra. {}'.format(time()-start))
    clustering3 = clustering(x_norm, nclusts3, seeds, logger, start)
    with open('clusterings/dump6001all.json', 'w') as file:
        json.dump(clustering3, file)
    logger.info('All clustered to 600 (second round). Time since start {}.'.
                format(time()-start))
    '''
    clustering_file = 'clusterings/dump8500all.json'
    clustering_dict = clustering_to_dict(clustering_file)
    logger.info('Read clustering file. Since start {:2.0f} sec'.
                format(time()-start))

    names = ['revenue', 'carbon', 'deadwood', 'ha']
    optims = clustering_to_optims(x_stack,
                                  x_norm,
                                  x_norm_stack,
                                  names,
                                  clustering_dict,
                                  opt,
                                  logger=logger,
                                  starttime=start)
    with open('optimizations/dumpall8300all.json', 'w') as file:
        json.dump(optims, file)
    logger.info('All optimized. Since start {:2.0f} sec'.format(time()-start))

    nclusts4 = range(9900, 20000, 200)
    logger.info('Still some more super extra. {}'.format(time()-start))
    clustering4 = clustering(x_norm, nclusts4, seeds, logger, start)
    with open('clusterings/dump20000all.json', 'w') as file:
        json.dump(clustering4, file)
    logger.info('All clustered to 20000. Time since start {}.'.
                format(time()-start))
