import gradutil
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
from ASF import ASF, NIMBUS


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
                      column wise
        x_norm_stack: same as x_stack but values normalized to 0-1
                      scale column wise
        ideal:        Ideal vector of the problem
        nadir:        Nadir vector of the problem
        '''
        revenue, carbon, deadwood, ha = gradutil.init_boreal()
        n_revenue = gradutil.nan_to_bau(revenue)
        n_carbon = gradutil.nan_to_bau(carbon)
        n_deadwood = gradutil.nan_to_bau(deadwood)
        n_ha = gradutil.nan_to_bau(ha)

        self.x = pd.concat((n_revenue, n_carbon, n_deadwood, n_ha), axis=1)
        self.x_stack = np.dstack((n_revenue, n_carbon, n_deadwood, n_ha))

        self.x_norm = gradutil.normalize(self.x.values)
        self.x_norm_stack = gradutil.normalize(self.x_stack)

        if stock_ideal:
            self.ideal = gradutil.ideal(False)
            self.nadir = gradutil.nadir(False)
        else:
            self.ideal, self.nadir = gradutil.calc_ideal_n_nadir(self.x_stack)

    def cluster(self, clustdata=None, outdata=None, nclust=50,
                seedn=1, verbose=0):
        ''' Clusters the given data using kmeans algorithm and forms
        the centers for the clustering with another given data.
        clustdata N x dim data used in clustering, if no data given
                  used normalized data from the boreal files
                  (self.x_norm)
        outdata   N x dim data used for assigning cluster centers after
                  getting clusters using clustdata, if no data given
                  uses normalized stacked boreal data
                  (self.x_norm_stack)
        nclust    Number of clusters, default 50
        seedn     Random seed (for clustering)
        verbose   verbosity of used kmeans algorithm,
                  default 0: 0 no output, 2 extensive output
        "Saves" variables xtoc, dist, weights and centers

        return centers, weights and xtoc
        '''
        if clustdata is None:
            clustdata = self.x_norm
        if outdata is None:
            outdata = self.x_norm_stack
        self.c, self.xtoc, self.dist = gradutil.cluster(clustdata,
                                                        nclust,
                                                        seedn,
                                                        verbose=verbose)
        self.weights = np.array([sum(self.xtoc == i)
                                 for i in range(nclust)])

        self.centers = np.array([outdata[self.xtoc == i].mean(axis=0)
                                 for i in range(nclust)])

        return self.centers, self.weights, self.xtoc

    def values(self, data=None, xtoc=None, model=None):
        ''' Gives numerical values for a solved model, corresponding
        data and xtoc vector.
        data  Data to calculate values corresponding to the variables
              of the model
        xtoc  Relation between clusters and units in data (if clusters
              used in modelling), default self.xtoc from clustering
              method
        model Model to read optimal variable values from
        Returns the numerical values of objectives
        '''
        if data is None:
            data = self.x_stack
        if xtoc is None:
            xtoc = self.xtoc
        if model is None:
            model = self.SF.model
        return gradutil.model_to_real_values(data, xtoc, model)


class Solver():

    def __init__(self, model, solver='cplex'):
        self.solver = solver
        self.model = model
        self.opt = SolverFactory(solver)

    def solve(self, output=False, keepfiles=False):
        return self.opt.solve(self.model, tee=output, keepfiles=keepfiles)


if __name__ == '__main__':
    from time import time
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    start = time()
    logger.info('Started')
    logger.info('Initializing...')
    kehys = ReferenceFrame()
    logger.info('Initialized. Time since start {:2.0f} sec'.
                format(time()-start))
    nclust = 150
    logger.info('Clustering...')
    kehys.cluster(nclust=nclust)
    logger.info('Clustered. Time since start {:2.0f} sec'.format(time()-start))

    ref = np.array((0, 0, kehys.ideal[2], 0))

    logger.info('Using ideal: {} and nadir: {}'.
                format(kehys.ideal, kehys.nadir))
    logger.info('Solving...')

    data = kehys.centers
    weights = kehys.weights/len(kehys.x_norm)
    ideal = kehys.ideal
    nadir = kehys.nadir
    solver_name = 'cplex'

    asf = ASF(ideal, nadir, ref, data, weights=weights, scalarization='asf')
    stom = ASF(ideal, nadir, ref, data, weights=weights, scalarization='stom')
    guess = ASF(ideal, nadir, ref, data, weights=weights,
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

# ========================== NIMBUS ====================================

    ''' The data must be non-normalized, so that the limit values are
    matching during the nimbus scalarization'''
    nimbus_centers = np.array([kehys.x_stack[kehys.xtoc == i].mean(axis=0)
                               for i in range(nclust)])
    nimbus_weights = kehys.weights

    ''' Lets set classification so that starting from the asf-result
    of the previous problem, the first objective should improve, the
    second detoriate to a 2.5e+05, the third stay the same and the
    fourth change freely'''
    nimbus1_ref = np.array((kehys.ideal[0],
                            2.5e+06,
                            asf_values[2],
                            kehys.nadir[3]))

    ''' The classes whose 'distance' to the Pareto front are to be
    minized, i.e. the objectives to improve as much as possible and
    the ones to improve to a limit'''
    minmax1 = np.array([0], dtype=int)

    ''' The classes whose values are to be kept the same.'''
    stay1 = np.array([2], dtype=int)

    ''' The classes whose values are to be deteriorated to a limit'''
    detoriate1 = np.array([1], dtype=int)

    nimbus1 = NIMBUS(ideal, nadir, nimbus1_ref, nimbus_centers, minmax1,
                     stay1, detoriate1, asf_values, weights=nimbus_weights)
    nimbus1_solver = Solver(nimbus1.model, solver=solver_name)
    nimbus1_solver.solve()  # output=True, keepfiles=True)
    nimbus1_values = kehys.values(model=nimbus1.model)

    logger.info('Solved 4/4.')

    logger.info('Optimization done. Time since start {:2.0f} sec'.
                format(time()-start))

    logger.info('ASF:\n{}'.format(asf_values))
    logger.info('STOM:\n{}'.format(stom_values))
    logger.info('GUESS:\n{}'.format(guess_values))

    logger.info('From ASF, the first objective should improve, the second detoriate to a 2.5e+05, the third stay the same and the fourth change freely:\n{}'.format(nimbus1_values))
