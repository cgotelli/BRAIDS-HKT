"""
Module to set up run time parameters for Clawpack.

The values set in the function setrun are then written out to data files
that will be read in by the Fortran code.

"""

 
import os
 


from subprocess import Popen
from argparse import ArgumentParser
from yaml import safe_load
from pathlib import Path
import numpy as np
from clawpack.clawutil.data import ClawRunData
from clawpack.geoclaw import fgmax_tools
from clawpack.geoclaw import fgout_tools


proj_dir = Path(__file__).parents[1]
with open(proj_dir / "hydraulic_configuration.yaml") as file:
    config      = safe_load(file)
    lake        = config["lake"]
    topo        = config["topo_files"]
    computation = config["computation"]
    gauges      = config['gauges']
    output      = config['output']
    rheology    = config['rheology']
    # Validation/development capability only.  This key is deliberately not
    # produced by AVAC4QGIS and its absence leaves the established
    # AVAC-to-WAVE shoreline coupling unchanged.
    hidden_moving_solid = config.get('_hidden_moving_solid', {})

#avid        ='None'
topo_dir    = proj_dir / "Topo"


def _whole_cell_count(lower, upper, cell_size, axis):
    """Validate a positive whole-cell span without truncating float roundoff."""
    span = float(upper) - float(lower)
    spacing = float(cell_size)
    if not np.isfinite(span) or not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"WAVE {axis}-domain bounds and cell size must be finite, with positive spacing.")
    raw_count = span / spacing
    if not np.isfinite(raw_count):
        raise ValueError(f"WAVE {axis}-domain cell count must be finite.")
    count = int(round(raw_count))
    # Match AVAC's whole-cell check. Prepared decimal-valued bounds can
    # differ by roundoff; hand-written genuinely fractional grids must fail.
    if span <= 0.0 or count < 1 or not np.isclose(raw_count, count, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"WAVE {axis}-domain span {span:.12g} is not a whole number of "
            f"{spacing:.12g} m computational cells."
        )
    return count


#------------------------------
def setrun(claw_pkg='geoclaw'):
#------------------------------

    """
    Define the parameters used for running Clawpack.

    INPUT:
        claw_pkg expected to be "geoclaw" for this setrun.

    OUTPUT:
        rundata - object of class ClawRunData

    """

    from clawpack.clawutil import data

    assert claw_pkg.lower() == 'geoclaw',  "Expected claw_pkg = 'geoclaw'"

    num_dim = 2
    rundata = data.ClawRunData(claw_pkg, num_dim)

    #------------------------------------------------------------------
    # GeoClaw specific parameters:
    #------------------------------------------------------------------
    rundata = setgeo(rundata)

    #------------------------------------------------------------------
    # Standard Clawpack parameters to be written to claw.data:
    #   (or to amr2ez.data for AMR)
    #------------------------------------------------------------------
    clawdata = rundata.clawdata  # initialized when rundata instantiated

    # Set single grid parameters first.
    # See below for AMR parameters.

    # ---------------
    # Spatial domain:
    # ---------------

    # Number of space dimensions:
    clawdata.num_dim = num_dim

    #Addons
    #import AddSetrun
    #import AddZoom
    
    # Lower and upper edge of computational domain:

    clawdata.lower[0] = lake['xmin']
    clawdata.upper[0] = lake['xmax']
    clawdata.lower[1] = lake['ymin']
    clawdata.upper[1] = lake['ymax'] 
	 
    # Number of grid cells: Coarsest grid
    nx = _whole_cell_count(lake['xmin'], lake['xmax'], computation['cell_size'], 'x')
    ny = _whole_cell_count(lake['ymin'], lake['ymax'], computation['cell_size'], 'y')
    clawdata.num_cells[0] = nx
    clawdata.num_cells[1] = ny

    # ---------------
    # Size of system:
    # ---------------

    # Number of equations in the system:
    clawdata.num_eqn = 3

    # Number of auxiliary variables in the aux array (initialized in setaux)
    clawdata.num_aux = 1

    # Index of aux array corresponding to capacity function, if there is one:
    clawdata.capa_index = 0

    # -------------
    # Initial time:
    # -------------

    clawdata.t0 =  computation['t_0']

    # Restart from checkpoint file of a previous run?
    # If restarting, t0 above should be from original run, and the
    # restart_file 'fort.chkNNNNN' specified below should be in 
    # the OUTDIR indicated in Makefile.

    clawdata.restart      = False            # True to restart from prior results
    clawdata.restart_file = 'fort.chk00006'  # File to use for restart data

    # -------------
    # Output times:
    #--------------

    # Specify at what times the results should be written to fort.q files.
    # Note that the time integration stops after the final output time.
    # The solution at initial time t0 is always written in addition.

    clawdata.output_style = 1

    if clawdata.output_style==1:
        # Output nout frames at equally spaced times up to tfinal:
        clawdata.num_output_times = computation['nb_simul']
        clawdata.tfinal           = computation['t_max'] #+computation['t_0']
        clawdata.output_t0        = True  # output at initial (or restart) time?

    elif clawdata.output_style == 2:
        # Specify a list of output times.
        clawdata.output_times = [0.5, 1.0]

    elif clawdata.output_style == 3:
        # Output every iout timesteps with a total of ntot time steps:
        clawdata.output_step_interval = 1
        clawdata.total_steps = 1
        clawdata.output_t0 = True
        
    clawdata.output_format         = output['output_format']      # 'ascii' or 'binary' 
    clawdata.output_q_components   = 'all'   # could be list such as [True,True]
    clawdata.output_aux_components = [True]  # could be list
    clawdata.output_aux_onlyonce   = False    # output aux arrays only at t0

    # ---------------------------------------------------
    # Verbosity of messages to screen during integration:
    # ---------------------------------------------------

    # The current t, dt, and cfl will be printed every time step
    # at AMR levels <= verbosity.  Set verbosity = 0 for no printing.
    #   (E.g. verbosity == 2 means print only on levels 1 and 2.)
    clawdata.verbosity = 1

    # --------------
    # Time stepping:
    # --------------

    # if dt_variable==1: variable time steps used based on cfl_desired,
    # if dt_variable==0: fixed time steps dt = dt_initial will always be used.
    clawdata.dt_variable = True

    # Initial time step for variable dt.
    # If dt_variable==0 then dt=dt_initial for all steps:
    clawdata.dt_initial = 0.2

    # Max time step to be allowed if variable dt used:
    clawdata.dt_max = 1e+99

    # Desired Courant number if variable dt used, and max to allow without
    # retaking step with a smaller dt:
    clawdata.cfl_desired = computation['cfl_target']
    clawdata.cfl_max     = computation['cfl_max']

    # Maximum number of time steps to allow between output times:
    clawdata.steps_max = computation['max_iter']

    # ------------------
    # Method to be used:
    # ------------------

    # Order of accuracy:  1 => Godunov,  2 => Lax-Wendroff plus limiters
    clawdata.order = 2
    
    # Use dimensional splitting? (not yet available for AMR)
    clawdata.dimensional_split = 'unsplit'
    
    # For unsplit method, transverse_waves can be 
    #  0 or 'none'      ==> donor cell (only normal solver used)
    #  1 or 'increment' ==> corner transport of waves
    #  2 or 'all'       ==> corner transport of 2nd order corrections too
    clawdata.transverse_waves = 2

    # Number of waves in the Riemann solution:
    clawdata.num_waves = 3
    
    # List of limiters to use for each wave family:  
    # Required:  len(limiter) == num_waves
    # Some options:
    #   0 or 'none'     ==> no limiter (Lax-Wendroff)
    #   1 or 'minmod'   ==> minmod
    #   2 or 'superbee' ==> superbee
    #   3 or 'mc'       ==> MC limiter
    #   4 or 'vanleer'  ==> van Leer
    clawdata.limiter = [computation['limiter']]*3

    clawdata.use_fwaves = True    # True ==> use f-wave version of algorithms
    
    # Source terms splitting:
    #   src_split == 0 or 'none'    ==> no source term (src routine never called)
    #   src_split == 1 or 'godunov' ==> Godunov (1st order) splitting used, 
    #   src_split == 2 or 'strang'  ==> Strang (2nd order) splitting used,  not recommended.
    clawdata.source_split = 'godunov'

    # --------------------
    # Boundary conditions:
    # --------------------

    # Number of ghost cells (usually 2)
    clawdata.num_ghost = 2

    # Choice of BCs at xlower and xupper:
    #   0 => user specified (must modify bcN.f to use this option)
    #   1 => extrapolation (non-reflecting outflow)
    #   2 => periodic (must specify this at both boundaries)
    #   3 => solid wall for systems where q(2) is normal velocity

    clawdata.bc_lower[0] = computation['boundary']
    clawdata.bc_upper[0] = computation['boundary']

    clawdata.bc_lower[1] = computation['boundary']
    clawdata.bc_upper[1] = computation['boundary']

    # Specify when checkpoint files should be created that can be
    # used to restart a computation.

    clawdata.checkpt_style = 0

    if clawdata.checkpt_style == 0:
        # Do not checkpoint at all
        pass

    elif np.abs(clawdata.checkpt_style) == 1:
        # Checkpoint only at tfinal.
        pass

    elif np.abs(clawdata.checkpt_style) == 2:
        # Specify a list of checkpoint times.  
        clawdata.checkpt_times = [0.1,0.15]

    elif np.abs(clawdata.checkpt_style) == 3:
        # Checkpoint every checkpt_interval timesteps (on Level 1)
        # and at the final time.
        clawdata.checkpt_interval = 5

    # ---------------
    # AMR parameters:
    # ---------------
    amrdata = rundata.amrdata

    # Patch size includes the two-cell halo. Larger single-level patches
    # reduce ghost and scheduling work while keeping the same physical mesh.
    # Preserve GeoClaw's established patch partitioning for multilevel AMR.
    amrdata.max1d = 250 if int(computation['refinement']) == 1 else 60

    # max number of refinement levels:
    amrdata.amr_levels_max = computation['refinement']

    # List of refinement ratios at each level (length at least mxnest-1)
    amrdata.refinement_ratios_x = [10,5,2]
    amrdata.refinement_ratios_y = [10,5,2]
    amrdata.refinement_ratios_t = [10,5,2]

    # Specify type of each aux variable in amrdata.auxtype.
    # This must be a list of length maux, each element of which is one of:
    #   'center',  'capacity', 'xleft', or 'yleft'  (see documentation).

    amrdata.aux_type = ['center','capacity','yleft']


    # Flag using refinement routine flag2refine rather than richardson error
    amrdata.flag_richardson = False    # use Richardson?
    amrdata.flag2refine     = True

    # steps to take on each level L between regriddings of level L+1:
    amrdata.regrid_interval = 3

    # width of buffer zone around flagged points:
    # (typically the same as regrid_interval so waves don't escape):
    amrdata.regrid_buffer_width  = 2

    # clustering alg. cutoff for (# flagged pts) / (total # of cells refined)
    # (closer to 1.0 => more small grids may be needed to cover flagged cells)
    amrdata.clustering_cutoff = 0.700000

    # print info about each regridding up to this level:
    amrdata.verbosity_regrid = 0  

    # -------
    # Gauges:
    # -------
    rundata.gaugedata.gauges = []
    # for gauges append lines of the form  [gaugeno, x, y, t1, t2]
    #rundata.gaugedata.gauges.append([1, 986061, 6553729, clawdata.t0, clawdata.tfinal])
    if gauges['gauge_recording']:
        for k in range(len(gauges)-1):
            gaugeno = str(k)
            x = gauges[gaugeno]['x']
            y = gauges[gaugeno]['y']
            print(f"* gauge {k+1}: x = {x:.2f} m and y = {y:.2f} m")
            rundata.gaugedata.gauges.append([k+1, x, y, 0., 1e10])


    #  ----- For developers ----- 
    # Toggle debugging print statements:
    amrdata.dprint = False      # print domain flags
    amrdata.eprint = False      # print err est flags
    amrdata.edebug = False      # even more err est flags
    amrdata.gprint = False      # grid bisection/clustering
    amrdata.nprint = False      # proper nesting output
    amrdata.pprint = False      # proj. of tagged points
    amrdata.rprint = False      # print regridding summary
    amrdata.sprint = False      # space/memory output
    amrdata.tprint = False      # time step reporting each level
    amrdata.uprint = False      # update/upbnd reporting
    
    # More AMR parameters can be set -- see the defaults in pyclaw/data.py

    # == setregions.data values ==
    regions = rundata.regiondata.regions
    # to specify regions of refinement append lines of the form
    #  [minlevel,maxlevel,t1,t2,x1,x2,y1,y2]
    #regions.append([1, 1, 0., 1.e10, 925720,927160., 6451140.,6452155.])
    
    # == fgout_grids.data values ==
    # NEW IN v5.9.0
    # Set rundata.fgout_data.fgout_grids to be a list of
    # objects of class clawpack.geoclaw.fgout_tools.FGoutGrid:
    fgout_grids = rundata.fgout_data.fgout_grids  # empty list initially

    fgout               = fgout_tools.FGoutGrid()
    fgout.fgno          = 1       # for listing the files see doc
    fgout.point_style   = 2       # will specify a 2d grid of points
    fgout.output_format = output['output_format']  # 4-byte, float32
    fgout.nx = nx
    fgout.ny = ny
    fgout.x1 = lake['xmin'] #+ dx_fine/2.  # specify edges (fgout pts will be cell centers)
    fgout.x2 = lake['xmax'] #- dx_fine/2.
    fgout.y1 = lake['ymin'] #+ dx_fine/2
    fgout.y2 = lake['ymax'] #- dx_fine/2.
    fgout.tstart = computation['t_0']
    fgout.tend   = computation['t_max']  
    fgout.nout   = computation['nb_simul']+1
    fgout_grids.append(fgout)    # written to fgout_grids.data

    ################
    # setprob.data #
    ################
    probdata = rundata.new_UserData(name='probdata',fname='setprob.data')
    # Water-only is the default. A source file must be explicitly requested.
    probdata.add_param('inflow_file', str(config.get('internal_inflow_file', 'none')),
                       'Optional shoreline source; none for hydraulic runs')

    return rundata
    # end of function setrun
    # ----------------------


#-------------------
def setgeo(rundata):
#-------------------
    """
    Set GeoClaw specific runtime parameters.
    For documentation see ....
    """

    try:
        geo_data = rundata.geo_data
    except:
        print("*** Error, this rundata has no geo_data attribute")
        raise AttributeError("Missing geo_data attribute")


    #import AddSetrun       
    # == Physics ==
    geo_data.gravity           = rheology['gravity']
    geo_data.coordinate_system = 1
    geo_data.earth_radius      = 6367.5e3

    # == Forcing Options
    geo_data.coriolis_forcing = False

    # == Algorithm and Initial Conditions ==
    geo_data.sea_level           = lake["water_level"]
    geo_data.dry_tolerance       = computation['dry_limit']
    geo_data.friction_forcing    = rheology['friction']
    geo_data.manning_coefficient = 1/np.array(rheology['Strickler'])
    geo_data.friction_depth      = rheology['friction_depth_limit']
    geo_data.manning_break      = [rheology['friction_break_elevation']]

    # Refinement settings
    refinement_data = rundata.refinement_data
    refinement_data.variable_dt_refinement_ratios = True
    refinement_data.wave_tolerance = rheology['wave_tolerance_flag']

    # == settopo.data values ==
    topo_data = rundata.topo_data
    # for topography, append lines of the form
    #    [topotype, minlevel, maxlevel, t1, t2, fname]
    #topo_data.topofiles.append([2, 1, 4, 0., 1.e10, 'newtopo_1m_lac.asc'])
    topo_data.topofiles.append([3, 1, 4, 0., 1.e10, str(topo_dir / lake['topography']) ])
    rundata.topo_data.topo_missing = topo['missing_value']

    # == setdtopo.data values ==
    # Hidden validation-only moving-solid path.  The plugin has no control for
    # this and normal case files contain no ``_hidden_moving_solid`` mapping,
    # so the production shoreline source remains the only active coupling.
    if hidden_moving_solid.get('enabled', False):
        dtopo_name = hidden_moving_solid.get('dtopo_file')
        if not dtopo_name:
            raise ValueError(
                "_hidden_moving_solid.enabled requires a dtopo_file"
            )
        dtopo_path = Path(dtopo_name)
        if not dtopo_path.is_absolute():
            dtopo_path = proj_dir / dtopo_path
        dtopo_path = dtopo_path.resolve()
        if not dtopo_path.is_file():
            raise FileNotFoundError(
                "Hidden moving-solid dtopography is unavailable: "
                f"{dtopo_path}"
            )
        dt_max_dtopo = float(hidden_moving_solid.get('dt_max_dtopo', 1.e99))
        if not np.isfinite(dt_max_dtopo) or dt_max_dtopo <= 0.0:
            raise ValueError(
                "_hidden_moving_solid.dt_max_dtopo must be positive and finite"
            )
        rundata.dtopo_data.dtopofiles.append([3, str(dtopo_path)])
        rundata.dtopo_data.dt_max_dtopo = dt_max_dtopo

    # == setqinit.data values ==
    #rundata.qinit_data.qinit_type = 1
    #rundata.qinit_data.qinitfiles = []
    #rundata.qinit_data.qinitfiles = []
    # for qinit perturbations, append lines of the form: (<= 1 allowed for now!)
    #   [minlev, maxlev, fname]
    #rundata.qinit_data.qinitfiles.append([1, 2, 'new_initial50cm_1940.xyz'])
    # from clawpack.geoclaw.data import ForceDry
    # force_dry = ForceDry()
    # force_dry.tend = 5*60.
    # force_dry.fname = 'masque_lac_sec.asc'
    # rundata.qinit_data.force_dry_list.append(force_dry)
    
    # == setqinit.data values ==
    
    rundata.qinit_data.qinit_type = 0
    rundata.qinit_data.qinitfiles = []
    # for qinit perturbations, append lines of the form: (<= 1 allowed for now!)
    #   [fname]
    
    # NEW feature to adjust sea level by dtopo:
    rundata.qinit_data.variable_eta_init = True
    
    if 1:
        # NEW feature to force dry land some locations below sea level:
        # (comment out if you want the onshore depression to be a lake)
        from clawpack.geoclaw.data import ForceDry
        force_dry = ForceDry()
        force_dry.tend = 0.5+computation['t_0']
        force_dry.fname = str(topo_dir / topo['mask_raster'])   #
        rundata.qinit_data.force_dry_list.append(force_dry)
 


    # == setfixedgrids.data values ==
    #fixedgrids = rundata.fixed_grid_data.fixedgrids
    # for fixed grids append lines of the form
    # [t1,t2,noutput,x1,x2,y1,y2,xpoints,ypoints,\
    #  ioutarrivaltimes,ioutsurfacemax]

    return rundata
    # end of function setgeo
    # ----------------------



if __name__ == '__main__':
    # Set up run-time parameters and write all data files.
    import sys
    rundata = setrun(*sys.argv[1:])
    rundata.write()
