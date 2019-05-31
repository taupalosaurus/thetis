"""
Routines for interpolating forcing fields for the 3D solver.
"""
from firedrake import *
import numpy as np
import thetis.timezone as timezone
import thetis.interpolation as interpolation
import thetis.coordsys as coordsys
from .log import *
import netCDF4
import thetis.physical_constants as physical_constants
import os


def compute_wind_stress(wind_u, wind_v, method='LargePond1981'):
    r"""
    Compute wind stress from atmospheric 10 m wind.

    wind stress is defined as

    .. math:
        tau_w = C_D \rho_{air} \|U_{10}\| U_{10}

    where :math:`C_D` is the drag coefficient, :math:`\rho_{air}` is the density of
    air, and :math:`U_{10}` is wind speed 10 m above the sea surface.

    In practice `C_D` depends on the wind speed.

    Two formulation are currently implemented:

    - "LargePond1981":
        Wind stress formulation by [1]
    - "SmithBanke1975":
        Wind stress formulation by [2]

    [1] Large and Pond (1981). Open Ocean Momentum Flux Measurements in
        Moderate to Strong Winds. Journal of Physical Oceanography,
        11(3):324-336.
        https://doi.org/10.1175/1520-0485(1981)011%3C0324:OOMFMI%3E2.0.CO;2
    [2] Smith and Banke (1975). Variation of the sea surface drag coefficient with
        wind speed. Q J R Meteorol Soc., 101(429):665-673.
        https://doi.org/10.1002/qj.49710142920

    :arg wind_u, wind_v: Wind u and v components as numpy arrays
    :kwarg method: Choose the stress formulation. Currently supports:
        'LargePond1981' (default) or 'SmithBanke1975'.
    :returns: (tau_x, tau_y) wind stress x and y components as numpy arrays
    """
    rho_air = float(physical_constants['rho_air'])
    wind_mag = np.hypot(wind_u, wind_v)
    if method == 'LargePond1981':
        CD_LOW = 1.2e-3
        C_D = np.ones_like(wind_u)*CD_LOW
        high_wind = wind_mag > 11.0
        C_D[high_wind] = 1.0e-3*(0.49 + 0.065*wind_mag[high_wind])
    elif method == 'SmithBanke1975':
        C_D = (0.63 + 0.066 * wind_mag)/1000.
    tau = C_D*rho_air*wind_mag
    tau_x = tau*wind_u
    tau_y = tau*wind_v
    return tau_x, tau_y


class ATMNetCDFTime(interpolation.NetCDFTimeParser):
    """
    A TimeParser class for reading WRF/NAM atmospheric forecast files.
    """
    def __init__(self, filename, max_duration=24.*3600., verbose=False):
        """
        :arg filename:
        :kwarg max_duration: Time span to read from each file (in secords,
            default one day). Forecast files are usually daily files that
            contain forecast for > 1 days.
        :kwarg bool verbose: Se True to print debug information.
        """
        super(ATMNetCDFTime, self).__init__(filename, time_variable_name='time')
        # NOTE these are daily forecast files, limit time steps to one day
        self.start_time = timezone.epoch_to_datetime(float(self.time_array[0]))
        self.end_time_raw = timezone.epoch_to_datetime(float(self.time_array[-1]))
        self.time_step = np.mean(np.diff(self.time_array))
        self.max_steps = int(max_duration / self.time_step)
        self.time_array = self.time_array[:self.max_steps]
        self.end_time = timezone.epoch_to_datetime(float(self.time_array[-1]))
        if verbose:
            print_output('Parsed file {:}'.format(filename))
            print_output('  Raw time span: {:} -> {:}'.format(self.start_time, self.end_time_raw))
            print_output('  Time step: {:} h'.format(self.time_step/3600.))
            print_output('  Restricting duration to {:} h -> keeping {:} steps'.format(max_duration/3600., self.max_steps))
            print_output('  New time span: {:} -> {:}'.format(self.start_time, self.end_time))


class ATMInterpolator(object):
    """
    Interpolates WRF/NAM atmospheric model data on 2D fields.
    """
    def __init__(self, function_space, wind_stress_field,
                 atm_pressure_field, to_latlon,
                 ncfile_pattern, init_date, target_coordsys, verbose=False):
        """
        :arg function_space: Target (scalar) :class:`FunctionSpace` object onto
            which data will be interpolated.
        :arg wind_stress_field: A 2D vector :class:`Function` where the output
            wind stress will be stored.
        :arg atm_pressure_field: A 2D scalar :class:`Function` where the output
            atmospheric pressure will be stored.
        :arg to_latlon: Python function that converts local mesh coordinates to
            latitude and longitude: 'lat, lon = to_latlon(x, y)'
        :arg ncfile_pattern: A file name pattern for reading the atmospheric
            model output files. E.g. 'forcings/nam_air.local.2006_*.nc'
        :arg init_date: A :class:`datetime` object that indicates the start
            date/time of the Thetis simulation. Must contain time zone. E.g.
            'datetime(2006, 5, 1, tzinfo=pytz.utc)'
        :arg target_coordsys: coordinate system in which the model grid is
            defined. This is used to rotate vectors to local coordinates.
        :kwarg bool verbose: Se True to print debug information.
        """
        self.function_space = function_space
        self.wind_stress_field = wind_stress_field
        self.atm_pressure_field = atm_pressure_field

        # construct interpolators
        self.grid_interpolator = interpolation.NetCDFLatLonInterpolator2d(self.function_space, to_latlon)
        self.reader = interpolation.NetCDFSpatialInterpolator(self.grid_interpolator, ['uwind', 'vwind', 'prmsl'])
        self.timesearch_obj = interpolation.NetCDFTimeSearch(ncfile_pattern, init_date, ATMNetCDFTime, verbose=verbose)
        self.time_interpolator = interpolation.LinearTimeInterpolator(self.timesearch_obj, self.reader)
        lon = self.grid_interpolator.mesh_lonlat[:, 0]
        lat = self.grid_interpolator.mesh_lonlat[:, 1]
        self.vect_rotator = coordsys.VectorCoordSysRotation(
            coordsys.LL_WGS84, target_coordsys, lon, lat)

    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time.

        Performs interpolation and updates the output wind stress and
        atmospheric pressure fields in place.

        :arg float time: Thetis simulation time in seconds.
        """
        lon_wind, lat_wind, prmsl = self.time_interpolator(time)
        u_wind, v_wind = self.vect_rotator(lon_wind, lat_wind)
        u_stress, v_stress = compute_wind_stress(u_wind, v_wind)
        self.wind_stress_field.dat.data_with_halos[:, 0] = u_stress
        self.wind_stress_field.dat.data_with_halos[:, 1] = v_stress
        self.atm_pressure_field.dat.data_with_halos[:] = prmsl


class SpatialInterpolatorNCOM3d(interpolation.SpatialInterpolator):
    """
    Spatial interpolator class for interpolatin NCOM ocean model 3D fields.
    """
    def __init__(self, function_space, to_latlon, grid_path):
        """
        :arg function_space: Target (scalar) :class:`FunctionSpace` object onto
            which data will be interpolated.
        :arg to_latlon: Python function that converts local mesh coordinates to
            latitude and longitude: 'lat, lon = to_latlon(x, y)'
        :arg grid_path: File path where the NCOM model grid files
            ('model_lat.nc', 'model_lon.nc', 'model_zm.nc') are located.
        """
        self.function_space = function_space
        self.grid_path = grid_path

        # construct local coordinates
        xyz = SpatialCoordinate(self.function_space.mesh())
        tmp_func = self.function_space.get_work_function()
        xyz_array = np.zeros((tmp_func.dat.data_with_halos.shape[0], 3))
        for i in range(3):
            tmp_func.interpolate(xyz[i])
            xyz_array[:, i] = tmp_func.dat.data_with_halos[:]
        self.function_space.restore_work_function(tmp_func)

        self.latlonz_array = np.zeros_like(xyz_array)
        lat, lon = to_latlon(xyz_array[:, 0], xyz_array[:, 1], positive_lon=True)
        self.latlonz_array[:, 0] = lat
        self.latlonz_array[:, 1] = lon
        self.latlonz_array[:, 2] = xyz_array[:, 2]

        self._initialized = False

    def _get_forcing_grid(self, filename, varname):
        """
        Helper function to load NCOM grid files.
        """
        v = None
        with netCDF4.Dataset(os.path.join(self.grid_path, filename), 'r') as ncfile:
            v = ncfile[varname][:]
        return v

    def _create_interpolator(self, ncfile):
        """
        Create a compact interpolator by finding the minimal necessary support
        """
        lat_full = self._get_forcing_grid('model_lat.nc', 'Lat')
        lon_full = self._get_forcing_grid('model_lon.nc', 'Long')
        x_ind = ncfile['X_Index'][:].astype(int)
        y_ind = ncfile['Y_Index'][:].astype(int)
        lon = lon_full[y_ind, :][:, x_ind]
        lat = lat_full[y_ind, :][:, x_ind]

        # find where data values are not defined
        varkey = None
        for k in ncfile.variables.keys():
            if k not in ['X_Index', 'Y_Index', 'level']:
                varkey = k
                break
        assert varkey is not None, 'Could not find variable in file'
        vals = ncfile[varkey][:]  # shape nz, lat, lon
        land_mask = np.all(vals.mask, axis=0)

        # build 2d mask
        mask_good_values = ~land_mask
        # neighborhood mask with bounding box
        mask_cover = np.zeros_like(mask_good_values)
        buffer = 0.2
        lat_min = self.latlonz_array[:, 0].min() - buffer
        lat_max = self.latlonz_array[:, 0].max() + buffer
        lon_min = self.latlonz_array[:, 1].min() - buffer
        lon_max = self.latlonz_array[:, 1].max() + buffer
        mask_cover[(lat >= lat_min)
                   * (lat <= lat_max)
                   * (lon >= lon_min)
                   * (lon <= lon_max)] = True
        mask_cover *= mask_good_values
        # include nearest valid neighbors
        # needed for nearest neighbor filling
        from scipy.spatial import cKDTree
        good_lat = lat[mask_good_values]
        good_lon = lon[mask_good_values]
        ll = np.vstack([good_lat.ravel(), good_lon.ravel()]).T
        dist, ix = cKDTree(ll).query(self.latlonz_array[:, :2])
        ix = np.unique(ix)
        ix = np.nonzero(mask_good_values.ravel())[0][ix]
        a, b = np.unravel_index(ix, lat.shape)
        mask_nn = np.zeros_like(mask_good_values)
        mask_nn[a, b] = True
        # final mask
        mask = mask_cover + mask_nn

        self.nodes = np.nonzero(mask.ravel())[0]
        self.ind_lat, self.ind_lon = np.unravel_index(self.nodes, lat.shape)

        # find 3d mask where data is not defined
        vals = vals[:, self.ind_lat, self.ind_lon]
        self.good_mask_3d = ~vals.mask

        lat_subset = lat[self.ind_lat, self.ind_lon]
        lon_subset = lon[self.ind_lat, self.ind_lon]

        assert len(lat_subset) > 0, 'rank {:} has no source lat points'
        assert len(lon_subset) > 0, 'rank {:} has no source lon points'

        # construct vertical grid
        zm = self._get_forcing_grid('model_zm.nc', 'zm')
        zm = zm[:, y_ind, :][:, :, x_ind]
        grid_z = zm[:, self.ind_lat, self.ind_lon]  # shape (nz, nlatlon)
        grid_z = grid_z.filled(-5000.)
        # nudge water surface higher for interpolation
        grid_z[0, :] = 1.5
        nz = grid_z.shape[0]

        # data shape is [nz, neta*nxi]
        grid_lat = np.tile(lat_subset, (nz, 1))[self.good_mask_3d]
        grid_lon = np.tile(lon_subset, (nz, 1))[self.good_mask_3d]
        grid_z = grid_z[self.good_mask_3d]
        if np.ma.isMaskedArray(grid_lat):
            grid_lat = grid_lat.filled(0.0)
        if np.ma.isMaskedArray(grid_lon):
            grid_lon = grid_lon.filled(0.0)
        if np.ma.isMaskedArray(grid_z):
            grid_z = grid_z.filled(0.0)
        grid_latlonz = np.vstack((grid_lat, grid_lon, grid_z)).T

        # building 3D interpolator, this can take a long time (minutes)
        print_output('Constructing 3D GridInterpolator...')
        self.interpolator = interpolation.GridInterpolator(
            grid_latlonz, self.latlonz_array,
            normalize=True, fill_mode='nearest', dont_raise=True
        )
        print_output('done.')
        self._initialized = True

    def interpolate(self, nc_filename, variable_list, itime):
        """
        Calls the interpolator object
        """
        with netCDF4.Dataset(nc_filename, 'r') as ncfile:
            if not self._initialized:
                self._create_interpolator(ncfile)
            output = []
            for var in variable_list:
                assert var in ncfile.variables
                # TODO generalize data dimensions, sniff from netcdf file
                grid_data = ncfile[var][:][:, self.ind_lat, self.ind_lon][self.good_mask_3d]
                data = self.interpolator(grid_data)
                output.append(data)
        return output


class NCOMInterpolator(object):
    """
    Interpolates NCOM model data on 3D fields.

    .. note::
        The following NCOM output files must be present:
        ./forcings/ncom/model_h.nc
        ./forcings/ncom/model_lat.nc
        ./forcings/ncom/model_ang.nc
        ./forcings/ncom/model_lon.nc
        ./forcings/ncom/model_zm.nc
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006042000.nc
        ./forcings/ncom/2006/t3d/t3d.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/t3d/t3d.glb8_2f_2006042000.nc
        ./forcings/ncom/2006/u3d/u3d.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/u3d/u3d.glb8_2f_2006042000.nc
        ./forcings/ncom/2006/v3d/v3d.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/v3d/v3d.glb8_2f_2006042000.nc
        ./forcings/ncom/2006/ssh/ssh.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/ssh/ssh.glb8_2f_2006042000.nc
    """
    def __init__(self, function_space, fields, field_names, field_fnstr,
                 to_latlon, basedir,
                 file_pattern, init_date, target_coordsys, verbose=False):
        """
        :arg function_space: Target (scalar) :class:`FunctionSpace` object onto
            which data will be interpolated.
        :arg fields: list of :class:`Function` objects where data will be
            stored.
        :arg field_names: List of netCDF variable names for the fields. E.g.
            ['Salinity', 'Temperature'].
        :arg field_fnstr: List of variables in netCDF file names. E.g.
            ['s3d', 't3d'].
        :arg to_latlon: Python function that converts local mesh coordinates to
            latitude and longitude: 'lat, lon = to_latlon(x, y)'
        :arg basedir: Root dir where NCOM files are stored.
            E.g. '/forcings/ncom'.
        :arg file_pattern: A file name pattern for reading the NCOM output
            files (excluding the basedir). E.g.
            {year:04d}/{fieldstr:}/{fieldstr:}.glb8_2f_{year:04d}{month:02d}{day:02d}00.nc'.
        :arg init_date: A :class:`datetime` object that indicates the start
            date/time of the Thetis simulation. Must contain time zone. E.g.
            'datetime(2006, 5, 1, tzinfo=pytz.utc)'
        :arg target_coordsys: coordinate system in which the model grid is
            defined. This is used to rotate vectors to local coordinates.
        :kwarg bool verbose: Se True to print debug information.
        """
        self.function_space = function_space
        for f in fields:
            assert f.function_space() == self.function_space, 'field \'{:}\' does not belong to given function space {:}.'.format(f.name(), self.function_space.name)
        assert len(fields) == len(field_names)
        assert len(fields) == len(field_fnstr)
        self.field_names = field_names
        self.fields = dict(zip(self.field_names, fields))

        # construct interpolators
        self.grid_interpolator = SpatialInterpolatorNCOM3d(self.function_space, to_latlon, basedir)
        # each field is in different file
        # construct time search and interp objects separately for each
        self.time_interpolator = {}
        for ncvarname, fnstr in zip(field_names, field_fnstr):
            r = interpolation.NetCDFSpatialInterpolator(self.grid_interpolator, [ncvarname])
            pat = file_pattern.replace('{fieldstr:}', fnstr)
            pat = os.path.join(basedir, pat)
            ts = interpolation.DailyFileTimeSearch(pat, init_date, verbose=verbose)
            ti = interpolation.LinearTimeInterpolator(ts, r)
            self.time_interpolator[ncvarname] = ti
        # construct velocity rotation object
        self.rotate_velocity = ('U_Velocity' in field_names
                                and 'V_Velocity' in field_names)
        self.scalar_field_names = list(self.field_names)
        if self.rotate_velocity:
            self.scalar_field_names.remove('U_Velocity')
            self.scalar_field_names.remove('V_Velocity')
            lat = self.grid_interpolator.latlonz_array[:, 0]
            lon = self.grid_interpolator.latlonz_array[:, 1]
            self.vect_rotator = coordsys.VectorCoordSysRotation(
                coordsys.LL_WGS84, target_coordsys, lon, lat)

    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time
        """
        if self.rotate_velocity:
            # water_u (meter/sec) = Eastward Water Velocity
            # water_v (meter/sec) = Northward Water Velocity
            lon_vel = self.time_interpolator['U_Velocity'](time)[0]
            lat_vel = self.time_interpolator['V_Velocity'](time)[0]
            u, v = self.vect_rotator(lon_vel, lat_vel)
            self.fields['U_Velocity'].dat.data_with_halos[:] = u
            self.fields['V_Velocity'].dat.data_with_halos[:] = v

        for fname in self.scalar_field_names:
            vals = self.time_interpolator[fname](time)[0]
            self.fields[fname].dat.data_with_halos[:] = vals
