# Atmospheric pressure test case
#
# Flow in a rectangular channel is forced by a spatially sinusoidal atmospheric
# pressure gradient until the solution reaches a steady state, in which the
# pressure gradient is balanced by the elevation gradient.
#
# The elevation is compared to an analytic solution derived from the balance of
# pressure and elevation gradients.
#
# A high bottom friction coefficient is used to suppress oscillations and reach
# the steady state quickly.
#
from thetis import *
import pytest
import numpy as np


@pytest.mark.parametrize("element_family", [
    'dg-dg', 'rt-dg', 'dg-cg', ])
@pytest.mark.parametrize("timestepper", [
    'cranknicolson', 'ssprk33', ])
def test_pressure_forcing(element_family, timestepper):
    order = 1

    lx = 10000
    ly = 10000
    area = lx*ly

    # Test case parameters
    rho0 = physical_constants['rho0']
    g = physical_constants['g_grav']
    A = 2.0
    mu_manning = Constant(1.0)

    # Simulation time
    t_end = 43200.

    eta_errs = []

    n_tests = 3
    ns = [2.0**(i+1) for i in range(n_tests)]
    if timestepper == 'cranknicolson':
        dts = [2400.0/(2**i) for i in range(n_tests)]
    else:
        dts = [20.0/(2**i) for i in range(n_tests)]

    for i in range(n_tests):
        nx = ns[i]
        ny = ns[i]
        dt = dts[i]
        mesh2d = RectangleMesh(nx, ny, lx, ly)
        x = SpatialCoordinate(mesh2d)

        atmos_pressure_expr = -rho0*g*A*cos(pi*x[0]/lx)*cos(pi*x[1]/ly)
        eta_expr = A*cos(pi*x[0]/lx)*cos(pi*x[1]/ly)

        # bathymetry
        P1 = FunctionSpace(mesh2d, "DG", 1)
        bathymetry = Function(P1, name='bathymetry')
        bathymetry.interpolate(Constant(5.0))

        # atmospheric pressure
        atmospheric_pressure = Function(P1, name='atmospheric_pressure')
        atmospheric_pressure.interpolate(atmos_pressure_expr)

        # --- create solver ---
        solverObj = solver2d.FlowSolver2d(mesh2d, bathymetry)
        solverObj.options.order = order
        solverObj.options.timestepper_type = timestepper
        solverObj.options.element_family = element_family
        solverObj.options.dt = dt
        solverObj.options.t_export = dt
        solverObj.options.t_end = t_end
        solverObj.options.no_exports = True
        solverObj.options.fields_to_export = ['uv_2d', 'elev_2d']
        solverObj.options.shallow_water_theta = 0.5
        solverObj.options.mu_manning = mu_manning
        solverObj.options.atmospheric_pressure = atmospheric_pressure
        solverObj.options.solver_parameters_sw = {
            'snes_type': 'newtonls',
            'snes_monitor': False,
            'ksp_type': 'gmres',
            'pc_type': 'fieldsplit',
        }

        solverObj.assign_initial_conditions(uv=Constant((1e-7, 0.)))
        solverObj.iterate()

        uv, eta = solverObj.fields.solution_2d.split()
        eta_ana = project(eta_expr, solverObj.function_spaces.H_2d)
        eta_errs.append(errornorm(eta_ana, eta)/np.sqrt(area))

    eta_errs = np.array(eta_errs)
    expected_order = order + 1
    assert(all(eta_errs[:-1]/eta_errs[1:] > 2.**expected_order*0.75))
    assert(eta_errs[0]/eta_errs[-1] > (2.**expected_order)**(len(eta_errs)-1)*0.75)
    print_output("PASSED")