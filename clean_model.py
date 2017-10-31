from __future__ import division
from pyomo.environ import ConcreteModel, Param, RangeSet, Var
from pyomo.environ import Objective, Constraint, NonNegativeIntegers
from pyomo.environ import Binary, maximize
from pyomo.opt import SolverFactory

import os
import pandas as pd

data_dir = os.path.join(os.getcwd(), '../boreal_data')

carbon = pd.read_csv(os.path.join(data_dir, 'Carbon_storage.csv'))
carbon_clean = carbon.dropna(axis=0, how='any')

model = ConcreteModel()

data = carbon_clean[:10]
model.n = Param(within=NonNegativeIntegers, initialize=len(data))
model.m = Param(within=NonNegativeIntegers, initialize=len(list(data)))

model.I = RangeSet(0, model.n-1)
model.J = RangeSet(0, model.m-1)

model.x = Var(model.I, model.J, domain=Binary, initialize=0.0)
for i in model.I:
    model.x[i, 0].value = 1.0

    
def c_init(model, i, j):
    return carbon_clean[:10].values[i, j]


model.c = Param(model.I, model.J, initialize=c_init)


def obj_fun(model):
    return sum(sum(model.x[i, j]*model.c[i, j]
                   for i in model.I)
               for j in model.J)

model.OBJ = Objective(rule=obj_fun, sense=maximize)


def constra(model, i):
    return sum(model.x[i, j] for j in model.J) == 1


model.Constraint1 = Constraint(model.I, rule=constra)

opt = SolverFactory('glpk')
res = opt.solve(model, tee=True)
model.x.display()
