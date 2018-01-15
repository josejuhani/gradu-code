from pyomo.environ import ConcreteModel, Param, RangeSet, Var
from pyomo.environ import Objective, Constraint, NonNegativeIntegers
from pyomo.environ import Binary, maximize
from pyomo.opt import SolverFactory

import os
import numpy as np


class BorealWeightedProblem(object):

    def __init__(self, data, weights=None, objective=True):
        if weights is None:
            weights = np.ones(len(data))
        if len(data) != len(weights):
            print("Data and weights don't match in length")
            exit(1)
        model = ConcreteModel()
        # Number of lines in data
        model.n = Param(within=NonNegativeIntegers,
                        initialize=np.shape(data)[0])
        # Number of columns in data
        model.m = Param(within=NonNegativeIntegers,
                        initialize=np.shape(data)[1])

        model.I = RangeSet(0, model.n-1)
        model.J = RangeSet(0, model.m-1)

        # Initialize all x_ij = 0.0, when j != 0, and all x_i0 = 1.0
        model.x = Var(model.I, model.J, domain=Binary, initialize=0)
        for i in model.I:
            model.x[i, 0].value = 1.0

        # Initialize c_ij from given data
        def c_init(model, i, j):
            return data[i, j]

        model.c = Param(model.I, model.J, initialize=c_init)

        # Initialize w_i from given parameter)
        def w_init(model, i):
            return weights[i]

        model.w = Param(model.I, initialize=w_init)

        if objective:
            model.OBJ = Objective(rule=self.obj_fun, sense=maximize)

        def const(model, i):
            ''' Constraint: Given line i has only one 1
            \sum_{i=1}^{n}x_{ij} = 1'''
            return sum(model.x[i, j] for j in model.J) == 1

        model.Constraint1 = Constraint(model.I, rule=const)

        self.model = model
        self._modelled = True

    def obj_fun(self, model, c=None, w=None):
        '''Objective function: Formulate problem as binary problem.
        \sum_{i=1}^{n} \sum_{j=1}^{m} w_{i}*c_{ij}*x_{ij}'''
        if c is None:
            c = model.c
        if w is None:
            w = model.w
        return np.sum((np.sum((np.multiply(model.x[i, j], c[i, j]*w[i])
                               for i in model.I),
                              axis=0)
                      for j in model.J),
                      axis=0)


if __name__ == '__main__':
    import pandas as pd

    data_dir = os.path.join(os.getcwd(), '../boreal_data')
    carbon = pd.read_csv(os.path.join(data_dir, 'Carbon_storage.csv'))
    # Removes all lines containing Nan-values
    carbon_clean = carbon.dropna(axis=0, how='any')
    # Let's take a bit smaller sample
    data = carbon_clean[:1000].values
    # Problem without any weight so use just ones as weights
    weights = np.ones(len(data))
    problem = BorealWeightedProblem(data, weights)
    opt = SolverFactory('glpk')
    res = opt.solve(problem.model, True)
    # problem.model.x.display()
