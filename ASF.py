from pyomo.environ import Objective, Param, minimize, NonNegativeIntegers
from pyomo.environ import RangeSet, Constraint, Var
from BorealWeights import BorealWeightedProblem
import numpy as np


class ASF(BorealWeightedProblem):

    def __init__(self, z_ideal, z_nadir, z_ref, data, scalarization='ASF',
                 weights=None, eps=-1, roo=0.1):
        if len(z_ideal) != len(z_nadir) or len(z_ideal) != len(z_ref):
            print("Length of given vectors don't match")
            return
        super().__init__(data, weights, False)
        model = self.model

        # Initialize ASF parameters
        model.k = Param(within=NonNegativeIntegers,
                        initialize=len(z_ideal))
        model.H = RangeSet(0, model.k-1)

        def init_ideal(model, h):
            return z_ideal[h]

        def init_nadir(model, h):
            return z_nadir[h]

        def init_utopia(model, h):
            return z_ideal[h] - eps

        def init_ref(model, h):
            return z_ref[h]

        model.roo = roo

        scalarization = scalarization.upper()

        if scalarization == 'GUESS':
            model.z1 = Param(model.H, initialize=init_nadir)
            model.z2 = Param(model.H, initialize=init_nadir)
            model.z3 = Param(model.H, initialize=init_ref)
        elif scalarization == 'STOM':
            model.z1 = Param(model.H, initialize=init_utopia)
            model.z2 = Param(model.H, initialize=init_ref)
            model.z3 = Param(model.H, initialize=init_utopia)
        else:  # scalarization == 'ASF'
            model.z1 = Param(model.H, initialize=init_ref)
            model.z2 = Param(model.H, initialize=init_nadir)
            model.z3 = Param(model.H, initialize=init_utopia)

        model.maximum = Var()

        def minmaxconst(model, h):
            ''' Constraint: The new "maximum" variable, that will be minimized
            in the optimization, must be greater than any of the original
            divisions used in original ASF formulation.'''
            return model.maximum >= \
                np.divide(np.subtract(self.obj_fun(model, data)[h],
                                      model.z1[h]),
                          model.z2[h] - model.z3[h])

        model.ConstraintMax = Constraint(model.H, rule=minmaxconst)

        def asf_fun(model):
            return model.maximum \
                + model.roo*sum([np.divide(self.obj_fun(model, data)[h],
                                           model.z2[h] - model.z3[h])
                                 for h in model.H])

        if hasattr(model, 'OBJ'):
            del model.OBJ  # Delete previous Objective to suppress warnings
        model.OBJ = Objective(rule=asf_fun, sense=minimize)

        self.model = model
        self._modelled = True


if __name__ == '__main__':
    from gradutil import init_boreal, nan_to_bau, \
        ideal, nadir, values_to_list, normalize
    revenue, carbon, deadwood, ha = init_boreal()
    ind = 10
    x = np.dstack((normalize(nan_to_bau(revenue[:ind]).values),
                   normalize(nan_to_bau(carbon[:ind]).values),
                   normalize(nan_to_bau(deadwood[:ind]).values),
                   normalize(nan_to_bau(ha[:ind]).values)))
    X = np.dstack((nan_to_bau(revenue[:ind]).values,
                   nan_to_bau(carbon[:ind]).values,
                   nan_to_bau(deadwood[:ind]).values,
                   nan_to_bau(ha[:ind]).values))
    data = nan_to_bau(revenue).values
    ide = ideal(False)
    nad = nadir(False)
    ref = np.array((0, 6, 3, 8))
    asf = ASF(ide, nad, ref, x)
    from pyomo.opt import SolverFactory
    opt = SolverFactory('cplex')
    opt.solve(asf.model, tee=True)
    print(np.sum(values_to_list(asf, x), axis=0))
