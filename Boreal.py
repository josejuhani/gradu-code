from pyomo.environ import ConcreteModel, Param, RangeSet, Var
from pyomo.environ import Objective, Constraint, NonNegativeIntegers
from pyomo.environ import Binary, maximize
from pyomo.opt import SolverFactory

import os
import pandas as pd


class BorealModel(object):

    def __init__(self, data):
        self._solved = False
        self.res = None
        model = ConcreteModel()
        # Number of lines in data
        model.n = Param(within=NonNegativeIntegers, initialize=len(data))
        # Number of columns in data
        model.m = Param(within=NonNegativeIntegers, initialize=len(list(data)))

        model.I = RangeSet(0, model.n-1)
        model.J = RangeSet(0, model.m-1)

        # Initialize all x_ij = 0.0, when j != 0, and all x_i0 = 1.0
        model.x = Var(model.I, model.J, domain=Binary, initialize=0.0)
        for i in model.I:
            model.x[i, 0].value = 1.0

        # Initialize c_ij from given data
        def c_init(model, i, j):
            return data.values[i, j]

        model.c = Param(model.I, model.J, initialize=c_init)

        '''Objective function: Formulate problem as binary problem.
        \sum_{i=1}^{n} \sum_{j=1}^{m} c_{ij}*x_{ij}'''
        def obj_fun(model):
            return sum(sum(model.x[i, j]*model.c[i, j]
                           for i in model.I)
                       for j in model.J)

        model.OBJ = Objective(rule=obj_fun, sense=maximize)

        ''' Constraint: Given line i has only one 1
        \sum_{i=1}^{n}x_{ij} = 1'''
        def const(model, i):
            return sum(model.x[i, j] for j in model.J) == 1

        model.Constraint1 = Constraint(model.I, rule=const)

        self.model = model
        self._modelled = True

    def solve_model(self, solver='glpk', tee=True):
        opt = SolverFactory(solver)
        self.res = opt.solve(self.model, tee=tee)
        self._solved = True


if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), '../boreal_data')
    carbon = pd.read_csv(os.path.join(data_dir, 'Carbon_storage.csv'))
    # Removes all lines containing Nan-values
    carbon_clean = carbon.dropna(axis=0, how='any')
    data = carbon_clean[:1000]
    solver = BorealModel(data)
    solver.solve_model()
    solver.model.x.display()
