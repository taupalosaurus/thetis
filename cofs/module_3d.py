"""
3D momentum and tracer equations for hydrostatic Boussinesq flow.

Tuomas Karna 2015-02-23
"""
from utility import *
from physical_constants import *

g_grav = physical_constants['g_grav']
wd_alpha = physical_constants['wd_alpha']
mu_manning = physical_constants['mu_manning']
z0_friction = physical_constants['z0_friction']
von_karman = physical_constants['von_karman']


class ForwardEuler(timeIntegrator):
    """Standard forward Euler time integration scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        RHS = self.equation.RHS
        Source = self.equation.RHS

        invdt = Constant(1.0/dt)

        self.solution_old = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            self.funcs_old[k] = Function(self.funcs[k].function_space())
        # assing values to old functions
        for k in self.funcs:
            self.funcs_old[k].assign(self.funcs[k])

        u = self.equation.solution
        u_old = self.solution_old
        u_tri = self.equation.tri

        #self.F = (invdt*massTerm(u) - invdt*massTerm(u_old) -
                  #RHS(u_old, **self.funcs_old))

        self.A = (invdt*massTerm(u_tri))
        self.L = (invdt*massTerm(u_old) +
                  RHS(u_old, **self.funcs_old) +
                  Source(**self.funcs_old))

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
        }
        if updateForcings is not None:
            updateForcings(t+dt/2)
        #for k in self.funcs:
            #self.funcs_old[k].assign(0.5*(self.funcs[k]+self.funcs_old[k]))

        #solve(self.F == 0, solution, solver_parameters=solver_parameters)
        solve(self.A == self.L, solution)
        # store old values
        for k in self.funcs:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)


class LeapFrogAM3(timeIntegrator):
    """Leap frog - Adams Moulton 3 predictor-corrector scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        RHS = self.equation.RHS
        Source = self.equation.Source

        invdt = Constant(1.0/dt)

        self.solution_nminushalf = Function(self.equation.space)
        self.solution_nplushalf = Function(self.equation.space)
        self.solution_nminusone = Function(self.equation.space)
        self.solution_n = Function(self.equation.space)

        # dict of all input functions at t_{n}
        self.funcs = self.equation.kwargs

        u = self.equation.solution
        u_tri = self.equation.tri

        self.gamma = 1.0/12.0
        self.A = (invdt*massTerm(u_tri))
        self.L_predict = (invdt*massTerm(self.solution_nminushalf) +
                          RHS(self.solution_n, **self.funcs) +
                          Source(**self.funcs))
        self.L_correct = (invdt*massTerm(self.solution_n) +
                          RHS(self.solution_nplushalf, **self.funcs) +
                          Source(**self.funcs))

    def predict(self, t, dt, solution, updateForcings):
        """Advances equations from t_{n-1/2} to t_{n+1/2}.
        RHS is evaluated at t_{n}."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
        }
        if updateForcings is not None:
            updateForcings(t)
        # all self.funcs_old need to be at t_{n}
        self.solution_nminushalf.assign((0.5-2*self.gamma)*self.solution_nminusone +
                                        (0.5+2*self.gamma)*self.solution_n)
        solve(self.A == self.L_predict, solution)
        self.solution_nplushalf.assign(solution)

    def correct(self, t, dt, solution, updateForcings):
        """Advances equations from t_{n} to t_{n+1}
        RHS is evaluated at t_{n+1/2}."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
        }
        if updateForcings is not None:
            updateForcings(t+dt/2)
        # all self.funcs_nplushalf need to be at t_{n+1/2}

        solve(self.A == self.L_correct, solution)
        # shift time
        self.solution_nminusone.assign(self.solution_n)
        self.solution_n.assign(solution)


class AdamsBashforth3(timeIntegrator):
    """Standard 3rd order Adams-Bashforth scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        RHS = self.equation.RHS
        Source = self.equation.Source

        self.solution_old = Function(self.equation.space)

        self.K1 = Function(self.equation.space)
        self.K2 = Function(self.equation.space)
        self.K3 = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            self.funcs_old[k] = Function(self.funcs[k].function_space())
        # assing values to old functions
        for k in self.funcs:
            self.funcs_old[k].assign(self.funcs[k])

        u = self.equation.solution
        u_old = self.solution_old
        u_tri = self.equation.tri

        # mass matrix for a linear equation
        self.A = massTermBasic(u_tri)
        self.RHS = RHS(u_old, **self.funcs_old) + Source(**self.funcs_old)

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        if updateForcings is not None:
            updateForcings(t+dt/2)
        #for k in self.funcs:
            #self.funcs_old[k].assign(0.5*(self.funcs[k]+self.funcs_old[k]))

        solve(self.A == self.RHS, self.K1)
        K1_mix = 23.0/12.0
        K2_mix = -4.0/3.0
        K3_mix = 5.0/12.0
        solution.assign(self.solution_old +
                        dt*K1_mix*self.K1 +
                        dt*K2_mix*self.K2 +
                        dt*K3_mix*self.K3)
        # shift tendencies for next time step
        self.K3.assign(self.K2)
        self.K2.assign(self.K1)
        # store old values
        for k in self.funcs:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)


class CrankNicolson(timeIntegrator):
    """Standard Crank-Nicolson time integration scheme."""
    def __init__(self, equation, dt, gamma=0.6):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        RHS = self.equation.RHS
        Source = self.equation.Source

        invdt = Constant(1.0/dt)

        self.solution_old = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                self.funcs_old[k] = Function(self.funcs[k].function_space())
        # assing values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

        u = self.equation.solution
        u_old = self.solution_old
        u_tri = self.equation.tri
        #Crank-Nicolson
        gamma_const = Constant(gamma)
        self.F = (invdt*massTerm(u) - invdt*massTerm(u_old) -
                  gamma_const*RHS(u, **self.funcs) -
                  gamma_const*Source(**self.funcs) -
                  (1-gamma_const)*RHS(u_old, **self.funcs_old) -
                  (1-gamma_const)*Source(**self.funcs_old))

        self.A = (invdt*massTerm(u_tri) -
                  gamma_const*RHS(u_tri, **self.funcs))
        self.L = (invdt*massTerm(u_old) + gamma_const*Source(**self.funcs) +
                  (1-gamma_const)*RHS(u_old, **self.funcs_old) +
                  (1-gamma_const)*Source(**self.funcs_old))

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        #solver_parameters = {
            #'snes_type': 'newtonls',
            #'snes_monitor': True,
        #}
        solver_parameters = {
            'snes_type': 'ksponly',
        }
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solution_old.assign(solution)
        #solve(self.F == 0, solution, solver_parameters=solver_parameters)
        solve(self.A == self.L, solution, solver_parameters=solver_parameters)
        # shift time
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])


class momentumEquation(equation):
    """3D momentum equation for hydrostatic Boussinesq flow."""
    def __init__(self, mesh, space, space_scalar, bnd_markers, bnd_len,
                 solution, eta, bathymetry, w=None,
                 uv_bottom=None, bottom_drag=None, viscosity_v=None,
                 nonlin=True):
        self.mesh = mesh
        self.space = space
        self.space_scalar = space_scalar
        self.nonlin = nonlin
        self.solution = solution
        # this dict holds all time dep. args to the equation
        self.kwargs = {'eta': eta,
                       'w': w,
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       'viscosity_v': viscosity_v,
                       }
        # time independent arg
        self.bathymetry = bathymetry

        # test and trial functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        self.horizontal_DG = self.space.ufl_element()._A.family() != 'Lagrange'
        self.vertical_DG = self.space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # boundary definitions
        self.boundary_markers = bnd_markers
        self.boundary_len = bnd_len

        ## compute lenth of all boundaries
        #self.boundary_len = {}
        #for i in self.boundary_markers:
            ##ds_restricted = Measure('ds', subdomain_id=int(i))
            ##one_func = Function(self.space_scalar).interpolate(Expression(1.0))
            ##self.boundary_len[i] = assemble(one_func * ds_restricted)
            ##print 'bnd', i, self.boundary_len[i]

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

    def getTimeStep(self, Umag=Constant(1.0)):
        csize = CellSize(self.mesh)
        H = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
        uu = TestFunction(H)
        grid_dt = TrialFunction(H)
        res = Function(H)
        a = uu * grid_dt * self.dx
        L = uu * csize / Umag * self.dx
        solve(a == L, res)
        return res

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx
        #return (solution[0]*self.test[0] + solution[1]*self.test[1]) * self.dx

    def massTermBasic(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx

    def RHS(self, solution, eta, w=None, viscosity_v=None,
            uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        # Advection term
        if self.nonlin:
            # in 3d: nabla_grad dot (u u)
            # weak form: nabla_grad(psi) : u u
            # NOTE not validated (term small in channel flow)
            Adv_h = -(Dx(self.test[0], 0)*solution[0]*solution[0] +
                      Dx(self.test[0], 1)*solution[0]*solution[1] +
                      Dx(self.test[1], 0)*solution[1]*solution[0] +
                      Dx(self.test[1], 1)*solution[1]*solution[1])
            F += Adv_h * self.dx
            if self.horizontal_DG:
                uv_rie = avg(solution)
                s = 0.5*(sign(uv_rie[0]*self.normal('-')[0] +
                              uv_rie[1]*self.normal('-')[1]) + 1.0)
                uv_up = solution('-')*s + solution('+')*(1-s)
                G += (uv_up[0]*uv_rie[0]*jump(self.test[0], self.normal[0]) +
                      uv_up[0]*uv_rie[1]*jump(self.test[0], self.normal[1]) +
                      uv_up[1]*uv_rie[0]*jump(self.test[1], self.normal[0]) +
                      uv_up[1]*uv_rie[1]*jump(self.test[1], self.normal[1]))*(self.dS_v)
                # Lax-Friedrichs stabilization
                gamma = Constant(0.05)
                G += gamma*dot(jump(self.test), jump(solution))*self.dS_v
            if self.vertical_DG:
                # NOTE bottom bnd doesn't work for DG vertical mesh
                G += (uv_rie[0]*uv_rie[0]*jump(self.test[0], self.normal[0]) +
                      uv_rie[0]*uv_rie[1]*jump(self.test[0], self.normal[1]) +
                      uv_rie[1]*uv_rie[0]*jump(self.test[1], self.normal[0]) +
                      uv_rie[1]*uv_rie[1]*jump(self.test[1], self.normal[1]))*(self.dS_h)
            G += (solution[0]*solution[0]*self.test[0]*self.normal[0] +
                  solution[0]*solution[1]*self.test[0]*self.normal[1] +
                  solution[1]*solution[0]*self.test[1]*self.normal[0] +
                  solution[1]*solution[1]*self.test[1]*self.normal[1])*(ds_t + ds_b)
            ## Vertical advection
            #Adv_v = -(Dx(self.test[0], 2)*solution[0]*w +
                      #Dx(self.test[1], 2)*solution[1]*w)
            #F += Adv_v * self.dx
            ##if self.horizontal_DG:
                ##w_rie = avg(w)
                ##uv_rie = avg(solution)
                ##G += (uv_rie[0]*w_rie*jump(self.test[0], self.normal[2]) +
                      ##uv_rie[1]*w_rie*jump(self.test[1], self.normal[2]))*self.dS_h

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary
                continue
                #if self.nonlin:
                    #G += (self.test[0]*solution[0]*solution[0]*self.normal[0] +
                          #self.test[0]*solution[0]*solution[1]*self.normal[1] +
                          #self.test[1]*solution[1]*solution[0]*self.normal[0] +
                          #self.test[1]*solution[1]*solution[1]*self.normal[1])*(ds_bnd)

            elif 'elev' in funcs:
                # prescribe elevation only
                h_ext = funcs['elev']
                uv_ext = solution
                t = self.normal[1] * self.e_x - self.normal[0] * self.e_y
                ut_in = dot(solution, t)
                # ut_ext = -dot(uv_ext,t) # assume zero
                un_in = dot(solution, self.normal)
                un_ext = dot(uv_ext, self.normal)

                if self.nonlin:
                    H = self.bathymetry + (eta + h_ext) / 2
                else:
                    H = self.bathymetry
                c_roe = sqrt(g_grav * H)
                un_riemann = dot(solution, self.normal) + c_roe / H * (eta - h_ext)/2
                H_riemann = H
                ut_riemann = tanh(4 * un_riemann / 0.02) * (ut_in)
                uv_riemann = un_riemann * self.normal + ut_riemann * t

                if self.nonlin:
                    G += un_riemann * dot(uv_riemann, self.test) * ds_bnd

        ## viscosity
        ## A double dot product of the stress tensor and grad(w).
        #K_momentum = -viscosity * (Dx(uv[0], 0) * Dx(self.U_test[0], 0) +
                                   #Dx(uv[0], 1) * Dx(self.U_test[0], 1) +
                                   #Dx(uv[1], 0) * Dx(self.U_test[1], 0) +
                                   #Dx(uv[1], 1) * Dx(self.U_test[1], 1))
        #K_momentum += viscosity/total_H*inner(dot(grad(total_H), grad(uv)),
                                              #self.U_test)
        #F -= K_momentum * self.dx

        # vertical viscosity
        if viscosity_v is not None:
            F += viscosity_v*(Dx(self.test[0], 2)*Dx(solution[0], 2) +
                              Dx(self.test[1], 2)*Dx(solution[1], 2))*dx
            if self.vertical_DG:
                raise NotImplementedError('Vertical diffusion has not been implemented for DG')
                # G += -viscosity_v * dot(psi, du/dz) * normal[2]
                # viscflux = viscosity_v*Dx(solution, 2)
                # G += -(avg(viscflux[0])*jump(self.test[0], normal[2]) +
                #        avg(viscflux[0])*jump(self.test[1], normal[2]))

        return -F - G

    def Source(self, eta, w=None, viscosity_v=None,
               uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms
        G = 0

        # external pressure gradient
        F += g_grav * inner(nabla_grad(eta), self.test) * self.dx

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary
                continue

            elif 'elev' in funcs:
                # prescribe elevation only
                h_ext = funcs['elev']
                G += g_grav * ((h_ext - eta) / 2) * \
                    inner(self.normal, self.test) * ds_bnd

            elif 'flux' in funcs:
                # prescribe normal volume flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_ext = funcs['flux'] / total_H / sect_len
                if self.nonlin:
                    G += un_ext*un_ext*inner(self.normal, self.test)*ds_bnd

        if viscosity_v is not None:
            # bottom friction
            if bottom_drag is not None and uv_bottom is not None:
                stress = bottom_drag*sqrt(uv_bottom[0]**2 +
                                          uv_bottom[1]**2)*uv_bottom
                BotFriction = (stress[0]*self.test[0] +
                               stress[1]*self.test[1])*ds_t
                F += BotFriction

        return -F


class verticalMomentumEquation(equation):
    """Vertical advection and diffusion terms of 3D momentum equation for
    hydrostatic Boussinesq flow."""
    def __init__(self, mesh, space, space_scalar, solution, w=None,
                 viscosity_v=None, uv_bottom=None, bottom_drag=None):
        self.mesh = mesh
        self.space = space
        self.space_scalar = space_scalar
        self.solution = solution
        # this dict holds all time dep. args to the equation
        self.kwargs = {'w': w,
                       'viscosity_v': viscosity_v,
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       }

        # test and trial functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        self.horizontal_DG = self.space.ufl_element()._A.family() != 'Lagrange'
        self.vertical_DG = self.space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

    def getTimeStep(self, Umag=Constant(1.0)):
        raise NotImplementedError('getTimeStep not implemented')

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx
        #return (solution[0]*self.test[0] + solution[1]*self.test[1]) * self.dx

    def massTermBasic(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx

    def RHS(self, solution, w=None, viscosity_v=None,
            uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the equations.
        Contains all terms that depend on the solution."""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        # Advection term
        if w is not None:
            # Vertical advection
            Adv_v = -(Dx(self.test[0], 2)*solution[0]*w +
                      Dx(self.test[1], 2)*solution[1]*w)
            F += Adv_v * self.dx
            if self.horizontal_DG:
                raise NotImplementedError('Adv term not implemented for DG')
                #w_rie = avg(w)
                #uv_rie = avg(solution)
                #G += (uv_rie[0]*w_rie*jump(self.test[0], self.normal[2]) +
                      #uv_rie[1]*w_rie*jump(self.test[1], self.normal[2]))*self.dS_h

        # vertical viscosity
        if viscosity_v is not None:
            F += viscosity_v*(Dx(self.test[0], 2)*Dx(solution[0], 2) +
                              Dx(self.test[1], 2)*Dx(solution[1], 2))*dx
            if self.vertical_DG:
                raise NotImplementedError('Vertical diffusion has not been implemented for DG')
                # G += -viscosity_v * dot(psi, du/dz) * normal[2]
                # viscflux = viscosity_v*Dx(solution, 2)
                # G += -(avg(viscflux[0])*jump(self.test[0], normal[2]) +
                #        avg(viscflux[0])*jump(self.test[1], normal[2]))

        return -F - G

    def Source(self, w=None, viscosity_v=None,
               uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms

        if viscosity_v is not None:
            # bottom friction
            if bottom_drag is not None and uv_bottom is not None:
                stress = bottom_drag*sqrt(uv_bottom[0]**2 +
                                          uv_bottom[1]**2)*uv_bottom
                BotFriction = (stress[0]*self.test[0] +
                               stress[1]*self.test[1])*ds_t
                F += BotFriction

        return -F


class tracerEquation(equation):
    """3D tracer advection-diffusion equation"""
    def __init__(self, mesh, space, solution, eta, uv, w,
                 bnd_markers, bnd_len, nonlin=True):
        self.mesh = mesh
        self.space = space
        # this dict holds all args to the equation (at current time step)
        self.solution = solution
        self.kwargs = {'eta': eta, 'uv': uv, 'w': w}
        # time independent arg

        # trial and test functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        self.horizontal_DG = self.space.ufl_element()._A.family() != 'Lagrange'
        self.vertical_DG = self.space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # boundary definitions
        self.boundary_markers = bnd_markers
        self.boundary_len = bnd_len

        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx

    def massTermBasic(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx

    def RHS(self, solution, eta, uv, w, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        # NOTE advection terms must be exactly as in 3d continuity equation
        # Horizontal advection term
        F += -solution*(uv[0]*Dx(self.test, 0) +
                        uv[1]*Dx(self.test, 1))*dx
        # Vertical advection term
        F += -solution*w*Dx(self.test, 2)*dx

        # Bottom/top impermeability boundary conditions
        G += +solution*(uv[0]*self.normal[0] +
                        uv[1]*self.normal[1])*self.test*(ds_t + ds_b)
        G += +solution*w*self.normal[2]*self.test*(ds_t + ds_b)

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary NOTE uv.n should be very close to 0
                G += solution*(self.normal[0]*uv[0] +
                               self.normal[1]*uv[1])*self.test*ds_bnd

            elif 'value' in funcs:
                # prescribe external tracer value
                c_ext = funcs['value']
                G += c_ext*(self.normal[0]*uv[0] +
                            self.normal[1]*uv[1])*self.test*ds_bnd

        return -F - G

    def Source(self, eta, uv, w, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms

        return -F
