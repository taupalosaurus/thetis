# Tuomas Karna 2015-03-03
from cofs import *
import math


def test_steady_state_channel(do_export=False):

    lx = 5e3
    ly = 1e3
    # we don't expect converge as the reference solution neglects the advection term
    mesh2d = RectangleMesh(5, 1, lx, ly)

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name="bathymetry")
    bathymetry_2d.assign(100.0)

    n = 20  # number of timesteps
    dt = 100.
    g = physical_constants['g_grav'].dat.data[0]
    f = g/lx  # linear friction coef.

    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d, order=1)
    solver_obj.options.nonlin = False
    solver_obj.options.t_export = dt
    solver_obj.options.t_end = n*dt
    solver_obj.options.no_exports = not do_export
    # NOTE had to set to something else than Cr-Ni, otherwise overriding below has no effect
    solver_obj.options.timestepper_type = 'forwardeuler'
    solver_obj.options.timer_labels = []
    solver_obj.options.lin_drag = f
    solver_obj.options.dt = dt

    # boundary conditions
    inflow_tag = 1
    outflow_tag = 2
    inflow_func = Function(p1_2d)
    inflow_func.interpolate(Expression(-1.0))  # NOTE negative into domain
    inflow_bc = {'un': inflow_func}
    outflow_func = Function(p1_2d)
    outflow_func.interpolate(Expression(0.0))
    outflow_bc = {'elev': outflow_func}
    solver_obj.bnd_functions['shallow_water'] = {inflow_tag: inflow_bc, outflow_tag: outflow_bc}
    parameters['quadrature_degree'] = 5

    solver_obj.create_equations()
    solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_package': 'mumps',
        'snes_monitor': False,
        'snes_type': 'newtonls'}
    # reinitialize the timestepper so we can set our own solver parameters and gamma
    # setting gamma to 1.0 converges faster to
    solver_obj.timestepper = timeintegrator.CrankNicolson(solver_obj.eq_sw, solver_obj.dt,
                                                          solver_parameters, gamma=1.0)
    solver_obj.assign_initial_conditions(uv_init=Expression(("1.0", "0.0")))

    solver_obj.iterate()

    uv, eta = solver_obj.fields.solution_2d.split()

    eta_ana = interpolate(Expression("1-x[0]/lx", lx=lx), p1_2d)
    area = lx*ly
    l2norm = errornorm(eta_ana, eta)/math.sqrt(area)
    rel_err = math.sqrt(l2norm/area)
    print rel_err
    assert(rel_err < 1e-3)
    print "PASSED"


if __name__ == '__main__':
    test_steady_state_channel(do_export=True)