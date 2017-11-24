from pyomo.environ import Objective, Param, minimize, NonNegativeIntegers
from pyomo.environ import RangeSet
from BorealWeights import BorealWeightedProblem


class ASF(BorealWeightedProblem):

    def __init__(self, z_ideal, z_nadir, z_ref, data,
                 weights=None, eps=-1, roo=2):
        super().__init__(data, weights)
        model = self.model
        model.ideal = Param(initialize=z_ideal)
        model.nadir = Param(initialize=z_nadir)
        model.utopia = Param(initialize=z_ideal - eps)
        model.ref = Param(initialize=z_ref)
        model.roo = roo

        # Initialize ASF parameters
        model.k = Param(within=NonNegativeIntegers,
                        initialize=len(z_ideal))

        model.H = RangeSet(1, model.k)

        # Integrate all four objectives in this..
        
        def obj_fun(model):
            A = (self.obj_fun(model) - model.ref)/(model.nadir-model.utopia)
            return A - model.roo
        del model.OBJ  # Delete previous Objective to suppress warnings
        model.OBJ = Objective(rule=obj_fun, sense=minimize)

        self.model = model
        self._modelled = True


if __name__ == '__main__':
    from gradutil import init_boreal, nan_to_bau, ideal, nadir, values_to_list
    revenue, _, _, _ = init_boreal()
    data = nan_to_bau(revenue).values
    ideal = ideal()['revenue']
    nadir = nadir()['revenue']
    ref = 100000000
    asf = ASF(ideal, nadir, ref, data)
    from pyomo.opt import SolverFactory
    opt = SolverFactory('glpk')
    opt.solve(asf.model)
    print(sum(values_to_list(asf, data)))
