import numpy as np
#todo add ml_depth as param
#todo run with older settings and then turn on rb rg and see if still ok.
###################################################
##          Switches                ##
###################################################
# see main file for what loadcase corresponds to - I've used it for different data types
loadcase        = 0
# krauss turner
kt_switch       = 1
#Biddle Clark Ice (from thesis)
bc_ice          = 0
# modified akin to petty et al
kt_ice          = 1
#f or conductivity term in heat balance of kt_ice- implemented mainly for testing
cond_on     =  1        #
# critical bulk richardson number (0.65) set to 0 to turn off.
rb 		        = 0.65
# critical gradient richardson number (0.25) set to 0 to turn off.
rg		        = 0.25
# this is for inertial internal wave dissipation stuff, all other scripts I've seen have this unlabeled so assume 0
ucon            = 0
# relaxation switch
ocean_relax_switch  =     0


# depth below which "advection" occurs (m), if using this make sure ad_i is passed to
# relaxation function in main
ad		             = 597.


diffusion_switch    = 1
change=                   1
temp_diff	= 1e-4*change 	# diffusion paramater for t and s
salt_diff   = temp_diff
oxy_diff	=temp_diff      # diffusion paramater for DO
vel_diff	= 5e-4*change 	# diffusion of velocity fields.


##############################################
##             Sea Ice Settings             ##
##############################################

# upper limit of sea ice concentration.
A_grow      =         0.95
# lower limit of sea ice concentration.
A_melt      =      0.
# initial sea ice concentration
A_0		    = A_melt
# minimum sea ic concentration
A_min       = 0
#starting ice thickness
h_i0        = 0.
# average bulk salinity of sea ice
S_ice		= 5.
# minimum ice thickness
h_ice_min	= 0.
# snow thickness
h_snow      =          0.2 
# divergence of ice concentration (v3 only)
Div_yr =                 -0.4 
# lateral vs basal melting paramater (1 for all basal )
R_b=                       0.8 

###################################################
##          Simulation Paramaters               ##
###################################################
dt		    =  100
# time-step increment for saving to file (units seconds/dt)
dt_save		= round(160000./dt)
dz		    =  3.0
# the number of days to run
days 		= 100.
# the depth to run
depth 		= 750.
# this is a bit deprecated as I've put a line to find the mixed layer of the initial profile in the main
ml_depth_0  = 3.
# minimum allowable mixed layer
ml_min      = 3.
# max ml
ml_max      = depth-dz
# latitude (degrees)
lat 		= 60.

maxiter     = days*8.64e4/dt
nz          = int(depth/dz)+1
z           = np.arange(0,int(nz))*dz
# time scale of relaxation
OR_days             = 0.25*365.
#OR_days_dw          = 11.5      #this is timescale of deepwater fom louise thesis

#convert the relaxation timescale from years to seconds
OR_d               = (OR_days*24.*60.*60.)/dt
OR_timescale              =1./OR_d



#convert ice divergence to seconds
dy=Div_yr
if dy==0:
    Div=0.
    print('rightyo')
else:
    Div=1./(Div_yr*365.*24.*60.*60./dt)

#if relaxation below a specific depth, ad, get the index of this depth

ad_i		= int(ad/dz)
f 		    = 2*7.29*10**(-5)*np.sin(lat*np.pi/180.)
ang 	    = -f*dt/2.	# angle for current rotation. angle equal to inertial rotation for


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


#use snow albedo if there is snow, otherwise use ice props.
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
    cond_ice	    = 2.04		# cond of ice
cond_snow	    = 0.31		# conductivity value for snow
Latent_sub	    = 2.834*10**6	#latent heat of sublimation
Latent_vapor	= 2.501*10**6	# vaporisation
Latent_fusion	= 3.340*10**5	#
P_atm		    = 101325.		# atmospheric pressure (kPa)
rho_ocean_ref	= 1026.		# ocean reference
rho_ice_ref	    = 930.		# ice density
rho_air_ref	    = 1.275		# air density
stef_boltz	    = 5.67*10**(-8)	#stefan boltzmann constant
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

