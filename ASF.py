from pyomo.environ import ConcreteModel, Param, RangeSet, Var
from pyomo.environ import Objective, Constraint, NonNegativeIntegers
from pyomo.environ import Binary, maximize


class ASF():

    def __init__(self, sub_model,  z_ideal, z_nadir, z_ref):
        if len(z_ideal) != len(z_nadir) or len(z_ideal) != len(z_ref):
            print("Length of given vectors don't match")
            return(0)
        eps = -.0001
        model = ConcreteModel()

        # Copy the sub_model parameters to this model

        # Initialize ASF parameters
        model.k = Param(within=NonNegativeIntegers,
                        initialize=len(z_ideal))

        model.H = RangeSet(1, model.k)

        def init_ideal(model, h):
            return z_ideal[h-1]

        model.ideal = Param(model.H, initialize=init_ideal)

        def init_nadir(model, h):
            return z_nadir[h-1]

        model.nadir = Param(model.H, initialize=z_nadir)

        def init_utopia(model, h):
            return z_ideal[h-1] - eps

        model.utopia = Param(model.H, initialize=z_ideal - eps)

        def init_ref(model, h):
            return z_ref[h-1]
        model.ref = Param(model.H, initialize=init_ref)

        def obj_fun(model):
            NotImplemented

        model.OBJ = Objective(rule=obj_fun, sense=maximize)

        self.model = model
        self._modelled = True
