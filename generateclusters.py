from pyomo.opt import SolverFactory
from BorealWeights import BorealWeightedProblem
import numpy as np
import pandas as pd
import gradutil as gu
import simplejson as json


def clustering_to_optims(x_orig, x_opt, name, clustering, opt, logger=None,
                         starttime=None):
    optims = dict()
    for nclust in sorted(clustering.keys()):
        n_optims = dict()
        for seedn in clustering[nclust].keys():
            xtoc = np.array(clustering[nclust][seedn]['xtoc'])
            dist = np.array(clustering[nclust][seedn]['dist'])
            w = np.array([sum(xtoc == i)
                          for i in range(eval(nclust))
                          if sum(xtoc == i) > 0])
            c_close = np.array([x_opt[np.argmin(dist[xtoc == i])]
                                for i in range(eval(nclust))
                                if len(dist[xtoc == i]) > 0])
            prob = BorealWeightedProblem(c_close, weights=w)
            res = opt.solve(prob.model)
            n_optims[seedn] = gu.model_to_real_values(x_orig,
                                                            prob.model,
                                                            xtoc)
            if logger:
                logger.info('Optimized {} clusters. Seed {}'.format(nclust,
                                                                    seedn))
            if starttime:
                logger.info('Since start {}s.'.format(time()-starttime))
        optims[nclust] = n_optims
        if logger:
            logger.info('Optimized {} clusters'.format(nclust))
        if starttime:
            logger.info('Since start {}s.'.format(time()-starttime))
        with open('optimizations/dump{}{}.json'.format(name,
                                                       nclust), 'w') as file:
            json.dump(n_optims, file)
    return optims


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
                logger.info('Since start {}s.'.format(time()-starttime))
        res[nclust] = res_clust
        if logger:
            logger.info('Clustered to {} clusters'.format(nclust))
        if starttime:
            logger.info('Since start {}s.'.format(time()-starttime))
        with open('clusterings/dump{}.json'.format(nclust), 'w') as file:
            json.dump(res_clust, file)
    return res


if __name__ == '__main__':
    import logging
    from time import time
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
    logger.info('Initialized. {}'.format(time()-start))
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
    import os
    dire = os.path.join(os.getcwd(), 'clusterings')
    with open(os.path.join(dire, 'dump8500all.json'), 'r') as file:
        clustering8500 = json.loads(file.read())
    logger.info('Read clustering file.{}'.format(time()-start))

    optims_revenue = clustering_to_optims(n_revenue.values,
                                          revenue_norm,
                                          'revenue',
                                          clustering8500,
                                          opt,
                                          logger=logger,
                                          starttime=start)
    with open('optimizations/dumprevenue8500all.json', 'w') as file:
        json.dump(optims_revenue, file)
    logger.info('Revenue optimized. {}'.format(time()-start))

    optims_carbon = clustering_to_optims(n_carbon.values,
                                         carbon_norm,
                                         'carbon',
                                         clustering8500,
                                         opt,
                                         logger=logger,
                                         starttime=start)
    with open('optimizations/dumpcarbon8500all.json', 'w') as file:
        json.dump(optims_carbon, file)
    logger.info('Carbon optimized. {}'.format(time()-start))

    optims_deadwood = clustering_to_optims(n_deadwood.values,
                                           deadwood_norm,
                                           'deadwood',
                                           clustering8500,
                                           opt,
                                           logger=logger,
                                           starttime=start)
    with open('optimizations/dumpdeadwood8500all.json', 'w') as file:
        json.dump(optims_deadwood, file)
    logger.info('Deadwood optimized. {}'.format(time()-start))

    optims_ha = clustering_to_optims(n_ha.values,
                                     ha_norm,
                                     'ha',
                                     clustering8500,
                                     opt,
                                     logger=logger,
                                     starttime=start)
    with open('optimizations/dumpha8500all.json', 'w') as file:
        json.dump(optims_ha, file)
    logger.info('HA optimized. {}'.format(time()-start))

    nclusts4 = range(9900, 20000, 200)
    logger.info('Still some more super extra. {}'.format(time()-start))
    clustering4 = clustering(x_norm, nclusts4, seeds, logger, start)
    with open('clusterings/dump20000all.json', 'w') as file:
        json.dump(clustering4, file)
    logger.info('All clustered to 20000. Time since start {}.'.
                format(time()-start))
