import gradutil as gu
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
from ASF import ASF, NIMBUS
from scipy.spatial.distance import euclidean


class ReferenceFrame():
    def __init__(self, stock_ideal=True):
        ''' Initialize the Boreal Forest problem. Reads the data from
        the files to the variables and normalizes the data to the 0-1
        scale. Sets ideal and nadir values either from the stock
        (stock_ideal) or calculates them (which takes about 15 min)
        Variables available after initialization:
        x:            original revenue, carbon, deadwood and ha values
                      in one 29666 x 28 array
        x_stack:      original revenue, carbon, deadwood and ha values
                      in one (stacked) 29666 x 7 x 4 array
        x_norm:       same as x but values normalized to 0-1 scale
        x_norm_stack: same as x_stack but values normalized to 0-1
        ideal:        Ideal vector of the problem
        nadir:        Nadir vector of the problem
        '''
        revenue, carbon, deadwood, ha = gu.init_boreal()
        n_revenue = gu.nan_to_bau(revenue)
        n_carbon = gu.nan_to_bau(carbon)
        n_deadwood = gu.nan_to_bau(deadwood)
        n_ha = gu.nan_to_bau(ha)

        norm_revenue = gu.new_normalize(n_revenue.values)
        norm_carbon = gu.new_normalize(n_carbon.values)
        norm_deadwood = gu.new_normalize(n_deadwood.values)
        norm_ha = gu.new_normalize(n_ha.values)

        self.x = pd.concat((n_revenue, n_carbon, n_deadwood, n_ha), axis=1)
        self.x_stack = np.dstack((n_revenue, n_carbon, n_deadwood, n_ha))

        lbounds = np.min(np.min(self.x_stack, axis=1), axis=0)
        ubounds = np.max(np.max(self.x_stack, axis=1), axis=0)
        self.limits = np.array((lbounds, ubounds-lbounds))

        self.x_norm = np.concatenate((norm_revenue, norm_carbon,
                                      norm_deadwood, norm_ha), axis=1)
        self.x_norm_stack = self.normalize_ref(self.x_stack)

        if stock_ideal:
            self.ideal = gu.ideal(False)
            self.nadir = gu.nadir(False)
        else:
            self.ideal, self.nadir = gu.calc_ideal_n_nadir(self.x_stack)

    def normalize_ref(self, ref):
        ''' Normalizes the given reference point with the same scaling
        that is used for the data that is used in the optimization and
        clustering also. Will NOT alter the given point itself'''
        new_ref = ref.copy()
        new_ref -= self.limits[0]
        with np.errstate(invalid='ignore'):
            new_ref = np.where(self.limits[1] != 0.,
                               new_ref / self.limits[1],
                               0)
        return new_ref

    def cluster(self, clustdata=None, optdata=None, outdata=None,
                nclust=50, seedn=2, verbose=0):
        ''' Clusters the given data using kmeans algorithm and forms
        the centers for the clustering with another given data.
        clustdata N x dim data used in clustering, if no data given
                  used normalized data from the boreal files
                  (self.x_norm)
        optdata   N x dim data used for assigning cluster centers after
                  getting clusters using clustdata, if no data given
                  uses normalized stacked boreal data
                  (self.x_norm_stack)
        outdata   N x dim data used for calculating the values of
                  optimization resutls aka. the final output data, if
                  no data given used stacked boreal data
                  (self.x_norm)
        nclust    Number of clusters, default 50
        seedn     Random seed (for clustering)
        verbose   verbosity of used kmeans algorithm,
                  default 0: 0 no output, 2 extensive output
        "Saves" variables xtoc, dist, weights and centers

        return centers, weights and xtoc
        '''
        if clustdata is None:
            clustdata = self.x_norm
        if optdata is None:
            optdata = self.x_norm_stack
        if outdata is None:
            outdata = self.x_stack
        self.c, self.xtoc, self.dist = gu.cluster(clustdata,
                                                  nclust,
                                                  seedn,
                                                  verbose=verbose)

        self.weights = np.array([sum(self.xtoc == i)
                                 for i in range(nclust)
                                 if sum(self.xtoc == i) > 0])

        indices = [min(
            np.array(range(len(self.xtoc)))[self.xtoc == i],
            key=lambda index: euclidean(clustdata[index],
                                        np.mean(clustdata[self.xtoc == i],
                                                axis=0)))
                   for i in range(nclust) if sum(self.xtoc == i) > 0]
        self.centers = optdata[indices]
        self.out_centers = outdata[indices]

        return self.c, self.xtoc, self.dist

    def values(self, data=None, weights=None, model=None):
        ''' Gives numerical values for a solved model, corresponding
        data, weights and xtoc vector.
        data  Data to calculate values corresponding to the variables
              of the model, default self.out_centers
        xtoc  Relation between clusters and units in data (if clusters
              used in modelling), default self.xtoc from clustering
              method
        model Model to read optimal variable values from
        Returns the numerical values of objectives
        '''
        if data is None:
            data = self.out_centers
        if weights is None:
            weights = self.weights
        if model is None:
            model = self.SF.model
        return gu.cluster_to_value(data, gu.res_to_list(model), weights)


class Solver():

    def __init__(self, model, solver='cplex'):
        self.solver = solver
        self.model = model
        self.opt = SolverFactory(solver)

    def solve(self, output=False, keepfiles=False):
        return self.opt.solve(self.model, tee=output, keepfiles=keepfiles)


if __name__ == '__main__':
    from time import time
    from datetime import timedelta
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    start = time()
    logger.info('Started')
    logger.info('Initializing...')
    kehys = ReferenceFrame()
    logger.info('Initialized. Time since start {}'.
                format(timedelta(seconds=int(time()-start))))
    nclust = 600
    seedn = 6
    logger.info('Clustering...')
    '''
    import simplejson as json
    with open('clusterings/new_300.json', 'r') as file:
        clustering = json.load(file)
    kehys.xtoc = np.array(clustering['5']['xtoc'])
    kehys.weights = np.array([sum(kehys.xtoc == i)
                              for i in range(nclust)
                              if sum(kehys.xtoc == i) > 0])
    '''
    kehys.cluster(nclust=nclust, seedn=seedn)
    logger.info('Clustered. Time since start {}'.
                format(timedelta(seconds=int(time()-start))))

    init_ref = np.array((0, 0, kehys.ideal[2], 0))
    ref = kehys.normalize_ref(init_ref)

    logger.info('Using ideal: {} and nadir: {}'.
                format(kehys.ideal, kehys.nadir))
    logger.info('Reference point: {}.'.format(init_ref))
    logger.info('Solving...')

    data = kehys.centers
    nvar = len(kehys.x_norm)
    weights = kehys.weights/nvar

    ''' Because everything is scaled, scale these too'''
    ideal = kehys.normalize_ref(kehys.ideal)
    nadir = kehys.normalize_ref(kehys.nadir)

    solver_name = 'cplex'

    asf = ASF(ideal, nadir, ref, data, weights=weights, nvar=nvar,
              scalarization='asf')
    stom = ASF(ideal, nadir, ref, data, weights=weights, nvar=nvar,
               scalarization='stom')
    guess = ASF(ideal, nadir, ref, data, weights=weights, nvar=nvar,
                scalarization='guess')

    asf_solver = Solver(asf.model, solver=solver_name)
    asf_solver.solve()
    logger.info('Solved 1/4.')

    stom_solver = Solver(stom.model, solver=solver_name)
    stom_solver.solve()
    logger.info('Solved 2/4.')

    guess_solver = Solver(guess.model, solver=solver_name)
    guess_solver.solve()
    logger.info('Solved 3/4.')

    asf_values = kehys.values(model=asf.model)
    stom_values = kehys.values(model=stom.model)
    guess_values = kehys.values(model=guess.model)

    logger.info('ASF:\n{}'.format(asf_values))
    logger.info('STOM:\n{}'.format(stom_values))
    logger.info('GUESS:\n{}'.format(guess_values))

# ========================== NIMBUS ====================================

    ''' Lets set classification so that starting from the asf-result
    of the previous problem, the first objective should improve, the
    second detoriate to a 2.5e+06, the third stay the same and the
    fourth change freely'''
    init_nimbus1_ref = np.array((kehys.ideal[0],
                                 2.5e+06,
                                 asf_values[2],
                                 kehys.nadir[3]))
    nimbus1_ref = kehys.normalize_ref(init_nimbus1_ref)
    ''' The classes whose 'distance' to the Pareto front are to be
    minized, i.e. the objectives to improve as much as possible and
    the ones to improve to a limit'''
    minmax1 = np.array([0], dtype=int)

    ''' The classes whose values are to be kept the same.'''
    stay1 = np.array([2], dtype=int)

    ''' The classes whose values are to be deteriorated to a limit'''
    detoriate1 = np.array([1], dtype=int)

    '''The current starting solution, scaled'''
    current = kehys.normalize_ref(asf_values)

    nimbus1 = NIMBUS(ideal, nadir, nimbus1_ref, data, minmax1,
                     stay1, detoriate1, current, weights=weights,
                     nvar=nvar)
    nimbus1_solver = Solver(nimbus1.model, solver=solver_name)
    nimbus1_solver.solve()  # output=True, keepfiles=True)
    nimbus1_values = kehys.values(model=nimbus1.model)

    logger.info('Solved 4/4.')

    logger.info('Optimization done. Time since start {}'.
                format(timedelta(seconds=int(time()-start))))

    logger.info("""From ASF, the first objective should improve,
    the second detoriate to a 2.5e+06,
    the third stay the same and the fourth change freely:
    {}""".format(nimbus1_values))
