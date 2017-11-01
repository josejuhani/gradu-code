from pyomo.environ import ConcreteModel, Param, RangeSet, Var
from pyomo.environ import Objective, Constraint, NonNegativeIntegers
from pyomo.environ import Binary, maximize
from pyomo.opt import SolverFactory

import os
import pandas as pd


class BorealSolver(object):

    def __init__(self, solver='glpk', tee=True):
        self.solver = solver
        self.tee = True
        self.x = None
        self.res = None

    def solveBoreal(self, data):
        model = ConcreteModel()
        model.n = Param(within=NonNegativeIntegers, initialize=len(data))
        model.m = Param(within=NonNegativeIntegers, initialize=len(list(data)))

        model.I = RangeSet(0, model.n-1)
        model.J = RangeSet(0, model.m-1)

        model.x = Var(model.I, model.J, domain=Binary, initialize=0.0)
        for i in model.I:
            model.x[i, 0].value = 1.0

        def c_init(model, i, j):
            return data.values[i, j]

        model.c = Param(model.I, model.J, initialize=c_init)

        def obj_fun(model):
            return sum(sum(model.x[i, j]*model.c[i, j]
                           for i in model.I)
                       for j in model.J)

        model.OBJ = Objective(rule=obj_fun, sense=maximize)

        def constra(model, i):
            return sum(model.x[i, j] for j in model.J) == 1

        model.Constraint1 = Constraint(model.I, rule=constra)

        opt = SolverFactory(self.solver)
        self.res = opt.solve(model, tee=self.tee)
        self.x = model.x


if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), '../boreal_data')
    carbon = pd.read_csv(os.path.join(data_dir, 'Carbon_storage.csv'))
    carbon_clean = carbon.dropna(axis=0, how='any')
    data = carbon_clean[:100]
    solver = BorealSolver()
    solver.solveBoreal(data)
    solver.x.display()
