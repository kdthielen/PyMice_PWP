import numpy as np
#todo add ml_depth as param
#todo run with older settings and then turn on rb rg and see if still ok.
###################################################
##          Switches                ##
###################################################
# for loadcase, 0 for matlab, 1 for summer, 2 for winter
loadcase        =                   0

kt_switch       = 1        # 1 krauss-turner on, anything else off

bc_ice          = 0        # if 1 turns on biddle-clark ice scheme

kt_ice          = 1

A_0		    = 0.6 		# initial sea ice concentration
A_grow      = 0.77
A_melt      = 0.6
A_min       = 0
h_i0        = 0.01
S_ice		= 5.		    # average bulk salinity of sea ice
h_ice_min	= 0.01		# minimum ice thickness
h_snow      =            0.2
Div_yr=                               0.8 
cond_on     =  1
R_b=                 0.25


rb 		        = 0#0.65		# critical bulk richardson number (0.65)
rg		        = 0#0.25		# critical gradient richardson number (0.25)
ucon            = 0          # this is for inertial internal wave dissipation stuff, all other scripts I've seen have this unlabeled so assume 0

ocean_relax_switch  = 1
OR_days             = 0.25*365.         #this is to maintain what it used to be from bc thesis.
OR_days_ml          = 0.25*365.
OR_days_dw          = 11.5

ad		             = 597.		# depth below which "advection" occurs (m)

diffusion_switch    = 1
change=                   1
temp_diff	= 1e-4*change #1.38*10**(-7)	# diffusion paramater for t and s
salt_diff   = temp_diff #6.9*10**(-10)
oxy_diff	=temp_diff #1*10**(-7)	# diffusion paramater for DO
vel_diff	= 5e-4*change  #1.83 * 10**(-6)		# diffusion of velocity fields.

###################################################
##          Simulation Paramaters               ##
###################################################

dt		    =                        100 
dt_save		= 1600 *100./dt		# time-step increment for saving to file (multiples of dt)
dz		    =                                    1.0 
days 		= 2000.     # the number of days to run
depth 		= 750.		# the depth to run
ml_depth_0  = 3.
ml_min      = 3.
ml_max      = depth-dz
lat 		= 65.		# latitude (degrees)

OR_d               = (OR_days*24.*60.*60.)/dt
OR_timescale              =1./OR_d

OR_d_dw               = (OR_days_dw*24.*60.*60.)/dt
OR_timescale_dw              =1./OR_d_dw

OR_d_ml               = (OR_days_ml*24.*60.*60.)/dt
OR_timescale_ml              =1./OR_d_ml


Div=1./(Div_yr*365.*24.*60.*60./dt)
ad_i		= int(ad/dz)
f 		    = 2*7.29*10**(-5)*np.sin(lat*np.pi/180.)
ang 	    = -f*dt/2.	# angle for current rotation. angle equal to inertial rotation for


print(Div)



###################################################
##                  Constants                    ##
###################################################

alpha		    = 5.82*10**(-5)	# thermal expansion coef K^(-1)
beta		    = 8*10**(-4)	# saline contraction coef
cp_air 		    = 1005.		# Specific heat cap of air (J kg^(-1) K^(-1))
cp_ocean	    = 4190.		#

snow_emiss	    = 0.99		#
snow_albedo	    = 0.8		#

ice_emiss       = 0.99      # EBERT
ice_albedo      = 0.63      # bitz1999
if h_snow>0.001:
    si_emiss=snow_emiss
    si_albedo=snow_albedo
else:
    si_emiss=ice_emiss
    si_albedo=ice_albedo

ocean_emiss	    = 0.97		#
ocean_albedo	= 0.06		#

g	 	        = 9.81		# gravitational acceleration
if cond_on == 0:
    cond_ice=0.
else:
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
I_0		        = 0.45		# fraction of shortwave that penetrates open water surface layer Niiler and Kraus 1977
beta1 		    = 0.6		# longwave extinction coefficient (0.6 m)
beta2		    = 20. 		# shortwave extinction coefficient (20 m)
ep		        = 0.62197  	# epsilon, ratio of molecular weight of water and dry air (0.622)
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

