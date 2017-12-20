import gradutil
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
from ASF import ASF


class ReferenceFrame():
    def __init__(self):
        ''' Initialize the Boreal Forest problem. Reads the data from
        the files to the variables and normalizes the data to the 0-1
        scale.
        Variabels available after initialization:
        x:            original revenue, carbon, deadwood and ha values
                      in one 29666 x 28 array
        x_stack:      original revenue, carbon, deadwood and ha values
                      in one (stacked) 29666 x 7 x 4 array
        x_norm:       same as x but values normalized to 0-1 scale
                      column wise
        x_norm_stack: same as x_stack but values normalized to 0-1
                      scale column wise
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

    def cluster(self, clustdata=None, outdata=None, nclust=50,
                seedn=1, verbose=0):
        ''' Clusters the given data using kmeans algorithm and forms the centers
        for the clustering with another given data.
        clustdata N x dim data used in clustering, if no data given used
                  normalized data from the boreal files (self.x_norm)
        outdata   N x dim data used for assigning cluster centers after getting
                  clusters using clustdata, if no data given uses normalized
                  stacked boreal data (self.x_norm_stack)
        nclust    Number of clusters, default 50
        seedn     Random seed (for clustering)
        verbose   verbosity of used kmeans algorithm, default 0: 0 no output,
                  2 extensive output
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

    def solve(self, ref, data=None, weights=None,
              ideal=None, nadir=None,
              scalarization='ASF', solver='glpk'):
        ''' Solve the scalarization problem using given reference point and additional
        parameters
        data          Data used in creating problem,
                      default self.centers (from clustering)
        weights       Weights for the data,
                      default self.weight (from clustering)
        ideal         Ideal point, default ideal for the original boreal data
        nadir         Nadir point, default nadir for the original boreal data
        scalarization Scalarization used to calculate point corresponding to
                      reference point, options 'STOM', 'GUESS' and 'ASF',
                      defaul 'ASF'
        solver        Solver used for solving the optimization problem,
                      default 'glpk'
        After runnig (self)SF.model formed and solved.
        Returns the problem class (ASF class)
        '''
        if data is None:
            data = self.centers
        if weights is None:
            weights = self.weights
        if ideal is None:
            ideal = gradutil.ideal(False)
        if nadir is None:
            nadir = gradutil.nadir(False)
        opt = SolverFactory(solver)
        self.SF = ASF(ideal, nadir, ref, data, weights=weights,
                      scalarization=scalarization)
        opt.solve(self.SF.model)
        return self.SF

    def values(self, data=None, xtoc=None, model=None):
        ''' Gives numerical values for a solved model, corresponding data and
        xtoc vector.
        data  Data to calculate values corresponding to the variables of the
              model
        xtoc  Relation between clusters and units in data (if clusters used in
              modelling), default self.xtoc from clustering method
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


if __name__ == '__main__':
    from gradutil import ideal
    ide = ideal(False)
    kehys = ReferenceFrame()
    kehys.cluster()
    ref = np.array((0, 0, ide[2], 0))
    kehys.solve(ref)
    print(kehys.values())
