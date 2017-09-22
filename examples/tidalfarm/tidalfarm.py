# Tidal farm optimisation example
# =======================================
#
# This example is based on the OpenTidalFarm example:
# http://opentidalfarm.readthedocs.io/en/latest/examples/headland-optimization/headland-optimization.html
#
# It optimises the layout of a tidalfarm using the so called continuous approach where
# the density of turbines within a farm (n/o turbines per unit area) is optimised. This
# allows a.o to include a cost term based on the number of turbines which is computed as
# the integral of the density. For more details, see:
#   S.W. Funke, S.C. Kramer, and M.D. Piggott, "Design optimisation and resource assessment
#   for tidal-stream renewable energy farms using a new continuous turbine approach",
#   Renewable Energy 99 (2016), pp. 1046-1061, http://doi.org/10.1016/j.renene.2016.07.039

# to enable a gradient-based optimisation using the adjoint to compute gradients,
# we need to import from thetis_adjoint instead of thetis
from thetis_adjoint import *
from numpy.random import rand
op2.init(log_level=INFO)

parameters['coffee'] = {}  # temporarily disable COFFEE due to bug

test_gradient = False  # whether to check the gradient computed by the adjoint
optimise = True

# setup the Thetis solver obj as usual:
mesh2d = Mesh('headland.msh')

tidal_amplitude = 5.
tidal_period = 12.42*60*60
H = 40
timestep = tidal_period/50

# create solver and set options
solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(H))
options = solver_obj.options
options.timestep = timestep
options.simulation_export_time = timestep
options.simulation_end_time = tidal_period/2
options.output_directory = 'outputs'
options.check_volume_conservation_2d = True
options.element_family = 'dg-dg'
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 0.6
options.timestepper_options.solver_parameters = {'snes_monitor': True,
                                                 'ksp_type': 'preonly',
                                                 'pc_type': 'lu',
                                                 'pc_factor_mat_solver_package': 'mumps',
                                                 'mat_type': 'aij'
                                                 }
options.horizontal_viscosity = Constant(100.0)
options.quadratic_drag_coefficient = Constant(0.0025)

# assign boundary conditions
left_tag = 1
right_tag = 2
coasts_tag = 3
tidal_elev = Function(FunctionSpace(mesh2d, "CG", 1), name='tidal_elev')
tidal_elev_bc = {'elev': tidal_elev}
# FIXME: adjoint of vectorial Constants is broken
noslip_bc = {'uv': Constant((0.0, 0.0))}
freeslip_bc = {'un': Constant(0.0)}
solver_obj.bnd_functions['shallow_water'] = {
    left_tag: tidal_elev_bc,
    right_tag: tidal_elev_bc,
    # coasts_tag: noslip_bc
    coasts_tag: freeslip_bc
}


# first setup all the usual SWE terms
solver_obj.create_equations()

# defines an additional turbine drag term to the SWE
turbine_friction = Function(FunctionSpace(mesh2d, "CG", 1), name='turbine_friction')


class TurbineDragTerm(shallowwater_eq.ShallowWaterMomentumTerm):
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.get_total_depth(eta_old)
        C_D = turbine_friction
        f = C_D * sqrt(dot(uv_old, uv_old)) * inner(self.u_test, uv) / total_h * self.dx(2)
        return -f


# add it to the shallow water equations
fs = solver_obj.fields.solution_2d.function_space()
u_test, eta_test = TestFunctions(fs)
u_space, eta_space = fs.split()
turbine_drag_term = TurbineDragTerm(u_test, u_space, eta_space,
                                    bathymetry=solver_obj.fields.bathymetry_2d,
                                    options=options)
solver_obj.eq_sw.add_term(turbine_drag_term, 'implicit')

turbine_friction.assign(0.0)
solver_obj.assign_initial_conditions(uv=as_vector((1e-7, 0.0)))

# Setup the functional. It computes a measure of the profit as the difference
# of the power output of the farm (the "revenue") minus the cost based on the number
# of turbines


# turbine characteristics:
C_T = 0.8  # turbine thrust coefficient
D_T = 16.  # turbine diameter
A_T = pi * (D_T/2)**2  # turbine cross section


class TurbineDiagnostics(DiagnosticCallback):
    name = 'turbine power'
    variable_names = ['current_power', 'avg_power', 'cost', 'current_profit', 'avg_profit']

    def _initialize(self):
        self.initial_val = None
        self._initialized = True
        # FIXME: should uv_2d, but split() doesn't work correctly in pyajdoint
        uv = self.solver_obj.fields.solution_2d

        # should multiply this by density to get power in W - assuming rho=1000 we get kW instead
        self.power_integral = turbine_friction * (uv[0]*uv[0] + uv[1]*uv[1])**1.5 * dx(2)

        # turbine friction=C_T*A_T/2.*turbine_density
        # cost integral is n/o turbines = \int turbine_density = \int c_t/(C_T A_T/2.)
        cost_integral = 1./(C_T*A_T/2.) * turbine_friction * dx(2)
        self.cost = assemble(cost_integral)

        self.break_even_wattage = 100  # (kW) amount of power produced per turbine on average to "break even" (cost = revenue)

        # we rescale the functional such that the gradients are ~ order magnitude 1.
        # the scaling is chosen such that the gradient of break_even_wattage * cost_integral is of order 1
        # the power-integral is assumed to be of the same order of magnitude
        self.scaling = 1./assemble(self.break_even_wattage/(C_T*A_T/2.) * dx(2, domain=mesh2d))

        self.dt = self.solver_obj.dt
        self.time_period = 0.
        self.total_energy = 0.
        self.functional_sum = 0.

    def __call__(self):
        if not hasattr(self, '_initialized') or self._initialized is False:
            self._initialize()
        self.time_period += float(self.dt)
        current_power = assemble(self.power_integral)
        self.total_energy += current_power * float(self.dt)
        avg_power = self.total_energy / self.time_period
        current_functional = self.scaling * (current_power - self.break_even_wattage * self.cost)
        self.avg_functional = self.scaling * (avg_power - self.break_even_wattage * self.cost)
        return (current_power, avg_power, self.cost, current_functional, self.avg_functional)

    def message_str(self, *args):
        line = 'Current: power: {}, cost: {}, functional: {}:\n'.format(args[0], args[2], args[3])
        line += 'Average: power: {}, cost: {}, functional: {}:'.format(args[1], args[2], args[4])
        return line


# a function to update the tidal_elev bc value every timestep
# we also use it to display the profit each time step (which will be a time-integrated into the functional)
x = SpatialCoordinate(mesh2d)
g = 9.81
omega = 2 * pi / tidal_period


def update_forcings(t):
    print_output("Updating tidal elevation at t = {}".format(t))
    tidal_elev.interpolate(tidal_amplitude*sin(omega*t + omega/pow(g*H, 0.5)*x[0]))


callback = TurbineDiagnostics(solver_obj)
solver_obj.add_callback(callback)

# run as normal (this run will be annotated by firedrake-adjoint)
solver_obj.iterate(update_forcings=update_forcings)


pause_annotation()

tfpvd = File('turbine_friction.pvd')


# our own version of a ReducedFunctional, which when asked
# to compute its derivative, calls the standard derivative()
# method of ReducedFunctional but additionaly outputs that
# gradient and the current value of the control to a .pvd
class MyReducedFunctional(ReducedFunctional):
    def derivative(self, **kwargs):
        dj = super(MyReducedFunctional, self).derivative(**kwargs)
        # need to make sure dj always has the same name in the output
        grad = dj.copy()
        grad.rename("Gradient")
        # same thing for the control
        tf = self.controls[0].data().copy()
        tf.rename('TurbineFriction')
        tfpvd.write(grad, tf)
        return dj


# this reduces the functional J(u, tf) to a function purely of
# rf(tf) = J(u(tf), tf) where the velocities u(tf) of the entire simulation
# are computed by replaying the forward model for any provided turbine friction tf
c = Control(turbine_friction)
rf = MyReducedFunctional(callback.avg_functional, c)

if test_gradient:
    # write out the taped annotation graph, which can be viewed with e.g. xdot
    get_working_tape().visualise('graph.dot', dot=True)
    # visualise initial gradient:
    dJdc = compute_gradient(callback.avg_functional, c)
    File('dJdc.pvd').write(dJdc)
    # setup a random direction in the function space along which to do the taylor test
    dJdc.vector()[:] = rand(dJdc.function_space().dim())
    minconv = taylor_test(rf, turbine_friction, dJdc)
    # if the gradient is computed correctly, we should get 2nd order convergence of the Taylor remainder
    print_output("Order of convergence with taylor test (should be 2) = {}".format(minconv))

    assert minconv > 1.95

if optimise:
    # compute maximum turbine density, keeping 2.5 x 5 diameters distance
    max_density = 1./(D_T * 2.5 * D_T * 5)
    max_tf = C_T * A_T/2. * max_density
    print_output("Maximum turbine density = {}".format(max_tf))

    tf_opt = maximise(rf, bounds=[0, max_tf],
                      options={'maxiter': 100})
