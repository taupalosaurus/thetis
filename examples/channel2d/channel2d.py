# 2D shallow water equations in a closed channel
# ==============================================
#
# Solves shallow water equations in closed rectangular domain
# with sloping bathymetry.
#
# Initially water elevation is set to a piecewise linear function
# with a slope in the deeper (left) end of the domain. This results
# in a wave that develops a shock as it reaches shallower end of the domain.
# This example tests the integrity of the 2D mode and stability of momentum
# advection.
#
# Setting
# solver_obj.nonlin = False
# uses linear wave equation instead, and no shock develops.
#
# Tuomas Karna 2015-03-03
from scipy.interpolate import interp1d
from thetis import *

outputdir = 'outputs'
mesh2d = Mesh('channel_mesh.msh')
print_output('Loaded mesh '+mesh2d.name)
print_output('Exporting to '+outputdir)
# total duration in seconds
t_end = 6 * 3600
# estimate of max advective velocity used to estimate time step
u_mag = Constant(6.0)
# export interval in seconds
t_export = 100.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')

depth_oce = 20.0
depth_riv = 5.0  # 5.0 closed
bath_x = np.array([0, 100e3])
bath_v = np.array([depth_oce, depth_riv])


def bath(x, y, z):
    padval = 1e20
    x0 = np.hstack(([-padval], bath_x, [padval]))
    vals0 = np.hstack(([bath_v[0]], bath_v, [bath_v[-1]]))
    return interp1d(x0, vals0)(x)


x_func = Function(P1_2d).interpolate(Expression('x[0]'))
bathymetry_2d.dat.data[:] = bath(x_func.dat.data, 0, 0)

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.t_export = t_export
options.t_end = t_end
options.outputdir = outputdir
options.u_advection = u_mag
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.timestepper_type = 'SSPRK33'
# options.timestepper_type = 'SSPIMEX'
# options.timestepper_type = 'CrankNicolson'
if options.timestepper_type in ['CrankNicolson', 'SSPIMEX']:
    options.dt = 10.0  # override estimated dt
# initial conditions, piecewise linear function
elev_x = np.array([0, 30e3, 100e3])
elev_v = np.array([6, 0, 0])


def elevation(x, y, z, x_array, val_array):
    padval = 1e20
    x0 = np.hstack(([-padval], x_array, [padval]))
    vals0 = np.hstack(([val_array[0]], val_array, [val_array[-1]]))
    return interp1d(x0, vals0)(x)


x_func = Function(P1_2d).interpolate(Expression('x[0]'))
elev_init = Function(P1_2d)
elev_init.dat.data[:] = elevation(x_func.dat.data, 0, 0,
                                  elev_x, elev_v)
solver_obj.assign_initial_conditions(elev=elev_init)

solver_obj.iterate()
