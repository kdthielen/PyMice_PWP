import numpy as np
###################################################
##          Switches                ##
###################################################

kt_switch       = 1         # 1 krauss-turner on, anything else off

bc_ice          = 1        # if 1 turns on biddle-clark ice scheme
A_0		    = 0  		# initial sea ice concentration
A_grow      = 0.77
A_melt      = 0.6
h_i0        = 0

rb 		        = 0.65		# critical bulk richardson number (0.65)
rg		        = 0.25		# critical gradient richardson number (0.25)
ucon            = 0          # this is for inertial internal wave dissipation stuff, all other scripts I've seen have this unlabeled so assume 0

ocean_relax_switch  = 1
ad		             = 597.		# depth below which "advection" occurs (m)

diffusion_switch    = 1
tracer_diff	= 0.001		# diffusion paramater for t and s
oxy_diff	= 0.001		# diffusion paramater for DO
vel_diff	= 0.005		# diffusion of velocity fields.

###################################################
##          Simulation Paramaters               ##
###################################################

dt		    = 100. 		# time-step increment (seconds)
dz		    = 3.		    # depth increment (meters)
days 		= 2000.		# the number of days to run
depth 		= 750.		# the depth to run

dt_save		= 1600		# time-step increment for saving to file (multiples of dt)
lat 		= 65.		# latitude (degrees)

S_ice		= 5.		    # average bulk salinity of sea ice

h_ice_min	= 0.		# minimum ice thickness
h_snow      = 0.2
A_max		= 0.95		#
R_b		    = 1.0         # 0 if all melt lateral, 1 if all basal probably depends on floe size distribution
phi_r		= 0 		# ridging influence on salt flux

ad_i		= int(ad/dz)
f 		    = 2*7.29*10**(-5)*np.sin(lat*np.pi/180.)
ang 	    = -f*dt/2.	# angle for current rotation. angle equal to inertial rotation for
T_si_0      = 0





###################################################
##                  Constants                    ##
###################################################

alpha		    = 5.82*10**(-5)	# thermal expansion coef K^(-1)
beta		    = 8*10**(-4)	# saline contraction coef
cp_air 		    = 1005.		# Specific heat cap of air (J kg^(-1) K^(-1))
cp_ocean	    = 4190.		#
ice_emiss	    = 1.		#
ocean_emiss	    = 0.97		#
g	 	        = 9.81		# gravitational acceleration
cond_ice	    = 2.04		#
cond_snow	    = 0.31		#
Latent_sub	    = 2.834*10**6	#
Latent_vapor	= 2.501*10**6	#
Latent_fusion	= 3.340*10**5	#
P_atm		    = 101325.		# atmospheric pressure (kPa)
rho_ocean_ref	= 1026.		#
rho_ice_ref	    = 930.		#
rho_air_ref	    =1.275		#
stef_boltz	    = 5.67*10**(-8)	#
snow_albedo	    = 0.8		#
ocean_albedo	= 0.0		#
I_0		        = 0.45		# fraction of shortwave that penetrates open water surface layer Niiler and Kraus 1977
beta1 		    = 0.6		# longwave extinction coefficient (0.6 m)
beta2		    = 20. 		# shortwave extinction coefficient (20 m)
ep		        = 0.62197  	# epsilon, ratio of molecular weight of water and dry air (0.622)
ice_albedo      = 0.9
c1		        = 0.8		# Tang 1991 max magnitude of wind stirring in ml
C_turb_i	    = 0.0013	# Ebert and Curry 1993 turbulent exchange over ice
C_turb_o	    = 0.001		# Ebert and Curry 1993 turbulent exchange over leads
Stanton		    = 0.006		# McPhee 1992 mixed layer to sea ice heat transfer
c_unsteady	    = 0.03		# Kim 1976
dw		        = 10.		# Lemke and Manley 1984 scale depth of dissipation
cm              = 0.03
m_kt		= 0.4		# Co-efficient for power provided by wind, in Kraus-Turner
n_kt		= 0.18		# Co-efficient for power provided by buoyancy, in K-T
cd_ocean	= 0.001		# drag coefficient ocean
cd_ice		= 0.0013	# drag coefficient of ice
rkz		    = 0.0001        # background diffusion (0)