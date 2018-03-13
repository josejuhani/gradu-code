from pyomo.environ import Objective, Param, minimize, NonNegativeIntegers
from pyomo.environ import RangeSet, Constraint, Var, Set
from BorealWeights import BorealWeightedProblem
import numpy as np


class ASF(BorealWeightedProblem):

    def __init__(self, z_ideal, z_nadir, z_ref, data, scalarization='ASF',
                 weights=None, nvar=None, eps=0.00001, roo=0.01):
        if len(z_ideal) != len(z_nadir) or len(z_ideal) != len(z_ref):
            print("Length of given vectors don't match")
            return

        super().__init__(data, weights, nvar, False)
        model = self.model

        # Initialize ASF parameters
        model.k = Param(within=NonNegativeIntegers,
                        initialize=len(z_ideal))
        model.H = RangeSet(0, model.k-1)

        def init_ideal(model, h):
            return z_ideal[h]

        def init_nadir(model, h):
            return z_nadir[h] - eps

        def init_utopia(model, h):
            return z_ideal[h] + eps

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
            divisions used in original problem formulation.'''
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


class NIMBUS(BorealWeightedProblem):

    def __init__(self, z_ideal, z_nadir, z_ref, data, to_minmax, to_stay,
                 to_detoriate, curr_values, weights=None, nvar=None,
                 eps=0.00001, roo=0.01):
        ''' Implements the NIMBUS method.
        Ideal, nadir and ref vectors of equal length
        data,         data without normalization
        to_minmax,    array of indices for the objectives to be improved
        to_stay,      array of indices for the objectives to stay the same
        to_detoriate, array of indices for the objectives to detoriate
                      to a limit
        curr_values,  current values of all the objectives as a vector'''
        if len(z_ideal) != len(z_nadir) or len(z_ideal) != len(z_ref):
            print("Length of given vectors don't match")
            return

        super().__init__(data, weights, False)

        model = self.model

        model.to_minmax = Set(within=NonNegativeIntegers, initialize=to_minmax)
        model.to_stay = Set(within=NonNegativeIntegers, initialize=to_stay)
        model.to_detoriate = Set(within=NonNegativeIntegers,
                                 initialize=to_detoriate)
        model.lim = Set(within=NonNegativeIntegers,
                        initialize=np.concatenate((to_minmax,
                                                   to_stay,
                                                   to_detoriate)))
        limits = np.zeros(len(z_ref))
        limits[to_minmax] = curr_values[to_minmax]
        limits[to_stay] = curr_values[to_stay]
        limits[to_detoriate] = z_ref[to_detoriate]

        model.k = Param(within=NonNegativeIntegers, initialize=len(z_ref))
        model.H = RangeSet(0, model.k-1)

        def init_ideal(model, h):
            return z_ideal[h]
        model.ideal = Param(model.H, initialize=init_ideal)

        def init_nadir(model, h):
            return z_nadir[h] - eps
        model.nadir = Param(model.H, initialize=init_nadir)

        def init_utopia(model, h):
            return z_ideal[h] + eps
        model.utopia = Param(model.H, initialize=init_utopia)

        def init_ref(model, h):
            return z_ref[h]
        model.ref = Param(model.H, initialize=init_ref)

        def init_limits(model, i):
            return limits[i]
        model.limits = Param(model.lim, initialize=init_limits)

        model.roo = roo
        model.maximum = Var()

        def nimbusconst_max(model, i):
            ''' Nimbus constraint: Set lower limits for all the objectives,
            because improving means now maximizing. Works when all the
            lower limits set properly as opposite to the Nimbus method'''
            return self.obj_fun(model, data)[i] >= model.limits[i]

        model.ConstraintNimbus = Constraint(model.lim, rule=nimbusconst_max)

        def minmaxconst(model, i):
            ''' Constraint: The new "maximum" variable, that will be minimized
            in the optimization, must be greater than any of the original
            divisions used in original ASF formulation.'''
            return model.maximum >= \
                np.divide(np.subtract(self.obj_fun(model, data)[i],
                                      model.ref[i]),
                          np.subtract(model.nadir[i], model.utopia[i]))

        model.ConstraintMax = Constraint(model.to_minmax, rule=minmaxconst)

        def asf_fun(model):
            return model.maximum \
                + model.roo*sum([np.divide(self.obj_fun(model, data)[h],
                                           model.nadir[h] - model.utopia[h])
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
