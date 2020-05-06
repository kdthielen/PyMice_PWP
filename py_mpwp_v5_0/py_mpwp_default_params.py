params = {
#todo run with older settings and then turn on rb rg and see if still ok.
###################################################
##          Switches                ##
###################################################
'kt_switch'       : 1 ,    # krauss turner
'bc_ice'          : 0 ,    #Biddle Clark Ice (from thesis)
'full_ice'        : 1 ,   # modified akin to petty et al
'ocean_relax_switch'  : 0 ,# relaxation switch
'ekman_switch'    : 1 ,    # ekman advection
'diffusion_switch'    : 1 ,# vertical diffusion
'rb' 		        : 0.65 , # critical bulk richardson number (0.65) set to 0 to turn off.
'rg'		        : 0.25 , # critical gradient richardson number (0.25) set to 0 to turn off.
#'ucon'            : 0 ,    # inertial internal wave dissipation stuff, all other scripts I've seen have this unlabeled so assume 0


###################################################
##          Simulation Paramaters               ##
###################################################
'dt'		    : 1200 ,   # time-step increment for saving to file (units seconds/dt)
'freq_save'	: 1.  ,    # number of output /day

# the number of days to run
'init_day'    : 22,
'ndays' 		: 225,

'dz'		    : 3.0 ,    # vertical resolution
'max_depth' 	: 750. ,   # the depth to run
'ml_min'      : 3. ,     # minimum allowable mixed layer

'lat' 		: 60. ,    # latitude (degrees)


###################################################
##               Input/output files              ##
###################################################
'output_file'          : 'output.nc',
'log_file'             : 'log.txt',
'forcing_file'         : 'forcing.nc',
'init_state_file'      : 'init_state.nc',


###################################################
##          physics                ##
###################################################
# depth below which "advection" occurs (m), if using this make sure ad_i is passed to
# relaxation function in main
'equation_of_state'   : 'eos-80' , # choose equation of state {'eos-80' or 'linear'}
'alpha0'      : 5.82e-5, # used only if equation_of_state='linear'
'beta0'     : 8e-4 ,
'rho_ocean_ref'	: 1026.	,	# ocean reference
'kappa'	    : 1e-5 ,     	# diffusion paramater for tracers
'kappa_u'  	: 5e-5  ,        # diffusion paramater for momentum
'w_ekman'     : 1.4e-6 ,

# time scale of relaxation
'OR_days'     : 180.,
#OR_days_dw          : 11.5      #this is timescale of deepwater fom louise thesis
'ad_depth'    : 597. , #if relaxation below a specific depth

# init sea ice model
'A_0'		    : 0.9 ,  # initial sea ice concentration
'h_i0'        : 0.5 ,  # starting ice thickness
'Div_yr'      : 0.5 ,  # divergence of ice concentration (v3 only)
'h_snow'      : 0.1 ,  # snow thickness

}