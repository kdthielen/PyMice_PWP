# ctrl f "todo" for things todo#
# todo set scalar save to not be just at end (if troubleshooting want that data)
# todo check switches
# todo import params change in modules
# todo add ice/snow albedo into penetrating flux.

# function for variable ice albedo
import numpy as np
import scipy.io as sio
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import time as tp
import datetime
from optparse import OptionParser
import os
from shutil import copyfile
from params import *
import pypwp_functions as pypwp

usage = """%prog output_dir data_fname param_name """

parser = OptionParser(usage)    #todo add input file names.
parser.add_option("--out",dest="output_dir",
                      default=None,
                      help="output directory filename, default is pypwpw_DDMMYY_HHMM")

parser.add_option("--fname",dest="data_fname",
                      default="profiles_",
                      help="name of data output file, default is data_*.txt")

parser.add_option("--init",dest="profile_input",
                      default="AS_LCDW.mat",
                      help="name of initial profile file (expects .mat type for now)")

parser.add_option("--force",dest="met_input",
                      default="AS_longmet.mat",
                      help="name of forcing file (expects .mat type for now)")

options, args = parser.parse_args()

if options.output_dir is None:
    date = datetime.datetime.now()
    simdate = str(date)[0:4] + str(date)[5:7] + str(date)[8:10] + '_' + str(date)[11:13] + str(date)[14:16]
    base_path = 'ham' #+ str(simdate)
    print("output directory not specified - using default: "+str(base_path))
else:
    base_path=str(options.output_dir)


filename=str(options.data_fname)




met_input_file=str(options.met_input)
profile_input_file=str(options.profile_input)




save_path =str(base_path)+'/data'
count=0
if not os.path.exists(base_path):
    os.makedirs(base_path)
else:
    temp_base_path=base_path
    while os.path.exists(temp_base_path):
        temp_base_path = base_path + '_' + str(count)
        count+=1
    base_path=temp_base_path
    os.makedirs(base_path)
print(base_path)
save_path = str(base_path) + '/data'
if not os.path.exists(save_path):
    os.makedirs(save_path)
copyfile('params.py',os.path.join(base_path,'params.py'))

print(base_path, filename, "param_copy.py")

###############################################
##  diagnostic plots (to add to)/modularize  ##
###############################################

def plot_preliminary(time, lw_force, sw_force, T_force,U_force):

    # print(OLR_ice, ILR_ice, ISW_ice, i_sens, i_lat, T_si)
    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    plt.title('' )
    plt.plot(time, lw_force, label='new')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel('lw')

    ax2 = fig.add_subplot(412)
    plt.plot(time, sw_force, label='new')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylabel('sw')

    ax3 = fig.add_subplot(413)
    plt.plot(time, T_force, label='new')
    plt.xlabel('time')
    plt.ylabel('T_force')

    ax4 = fig.add_subplot(414)
    plt.plot(time, U_force, label='new')
    plt.xlabel('time')
    plt.ylabel('U_force')
    plt.show()
    plt.clf()
    return 0

################################################# some things here may not be needed with certain switches (mr) but
##   Initialize simulation paramaters/arrays   ## setting them to 0 here and setting the save means no other adjustment is needed
################################################# besides setting the switch. not huge cost and fewer ifs.

maxiter     = days*8.64e4/dt
nz          = int(depth/dz)+1
z           = np.arange(0,int(nz))*dz
temp        = np.zeros(nz)
salt        = np.zeros(nz)
oxy         = np.zeros(nz)

# Initiate ice/ml

h_i         = h_i0
A           = A_0
ml_depth    = (dz*ml_depth_0)//dz
ml_index    = int(round(ml_depth_0/dz))
##  Initialize scalar arrays  ##

mld_save=[]
h_i_save=[]
we_save=[]
mr_save=[]
pb_save=[]
pw_save=[]
Bo_save=[]
A_save=[]
tml=[]
sml=[]
we  = 0
Pb  = 0
Pw  = 0
Bo  = 0
mr  = 0
A=A_0

u = np.zeros(nz)
v = np.zeros(nz)

absrb = pypwp.absorb(beta1,beta2,nz,dz)


##########################################
##   LOAD FORCING AND INITIAL PROFILE   ##
##########################################

print('loading ', str(met_input_file))

met_forcing = sio.loadmat(met_input_file)

# import forcing data
time_series = met_forcing['met']['time'][0,0][:,0]
T_a_series = met_forcing['met']['tair'][0,0][:,0]
lw_series = met_forcing['met']['lw'][0,0][:,0]
sw_series = met_forcing['met']['sw'][0,0][:,0]
shum_series = met_forcing['met']['shum'][0,0][:,0]
precip_series = met_forcing['met']['precip'][0,0][:,0]
U_a_series = met_forcing['met']['U'][0,0][:,0]
tx_series = met_forcing['met']['tx'][0,0][:,0]
ty_series = met_forcing['met']['ty'][0,0][:,0]


time=np.arange(0,maxiter)*dt/8.64e4+time_series[0]
nmet 	= days*8.64E4/dt
if time[-1] > time_series[-1]:      # check that length of forcing time> sim time otherwise warn and shorten.
    time = time[time<time_series[-1]]
    maxiter = len(time)
    maxiter = len(time)
    print('Met input shorter than # of days selected, truncating run to ', nmet*dt/8.64E4, ' day(s)')


#  Check the time-resolution of the inertial period and warn - dont have this as check not sure if wanted.
if dt > 1./10.*2.*3.14/f:
    print('Time step, dt, too large to accurately resolve the inertial period. Is this okay? (y/n)')


# interpolate forcing data on to the simulation timegrid
T_a_force	= griddata(time_series,T_a_series,time)
lw_force    = griddata(time_series,lw_series,time)
sw_force    = griddata(time_series,sw_series,time)
shum_force  = griddata(time_series,shum_series,time)
precip_force= griddata(time_series,precip_series,time)
U_a_force   = griddata(time_series,U_a_series,time)
tx_force    = griddata(time_series,tx_series,time)
ty_force    = griddata(time_series,ty_series,time)


#  -- Load initial t,s profile data. --
print('loading', str(profile_input_file))

initial_profile=sio.loadmat(profile_input_file)
initial_z=initial_profile['profile']['z'][0,0][0,:]
initial_salt=initial_profile['profile']['s'][0,0][0,:]
initial_temp=initial_profile['profile']['t'][0,0][0,:]
initial_oxy=initial_profile['profile']['oxy'][0,0][0,:]
initial_density=initial_profile['profile']['d'][0,0][0,:]


# Check depth domain of initial profile and truncate to deepest point of observations if shorter
if depth > initial_z[-1]:
    depth = initial_z[-1]
    nz = depth / dz
    z = np.arange(0, nz) * dz
    print('Profile input shorter than depth selected, truncating to', str(depth), ' meters')


#  -- Interpolate the profile variables at dz resolution. --
temp	= griddata(initial_z,initial_temp,z)
salt	= griddata(initial_z,initial_salt,z)
density	= griddata(initial_z,initial_density,z)
oxy = griddata(initial_z,initial_oxy,z)


#convert oxygen into umol/kg
oxy = oxy*44.658
oxy = oxy/density
oxy = oxy*1000

#copy original profiles for the ocean relaxation scheme (below ad_i)
temp_orig=temp.copy()
salt_orig=salt.copy()
oxy_orig=oxy.copy()
density=pypwp.density_0(temp_orig,salt_orig)

if h_snow > 0.001:
    si_albedo = snow_albedo
else:
    si_albedo = ice_albedo

# forcing_plots
#plot_preliminary(time,lw_force,sw_force,T_a_force,U_a_force)

############################################################
##     END SETUP AND INITIALIZATION: START SIMULATION     ##
############################################################

iteration=0
start = tp.time() # for timing purposes can comment out but costs negligible.
while iteration<maxiter:
    ##  Load Forcings data for time step  ##
    lw = lw_force[iteration]
    sw = sw_force[iteration]
    T_a = T_a_force[iteration]
    U_a = U_a_force[iteration]
    sp_hum = shum_force[iteration]
    precip = precip_force[iteration]
    tx = tx_force[iteration]
    ty = ty_force[iteration]



    ##  Relaxes to initial profile below certain depth (ad_i) - rudimentary 3d/2d paramaterization

    if ocean_relax_switch==1:
        temp,salt,oxy = pypwp.Ocean_relax(temp, salt, oxy, temp_orig, salt_orig, oxy_orig, ml_index,OR_timescale) # try with ad_i todo

    ##  diffusion - at the moment t and s diffuse at same rate - copied from mPWP (Biddle-Clark)
    if diffusion_switch==1:
        temp,salt,oxy,u,v = pypwp.diffusion(temp,salt,oxy,u,v,temp_diff,salt_diff,oxy_diff,vel_diff,nz)
        
    T_so = temp[0]

    ##  diffusion can change T/S values so recalc density
    density = pypwp.density_0(temp, salt)

    ############################
    ##    OCEAN HEAT BUDGET   ##
    ############################

    ##  radiation terms
    OLR 	= pypwp.lw_emission(T_so,ocean_emiss)
    ILR 	= pypwp.lw_downwelling(lw,ocean_emiss)
    ISW 	= pypwp.sw_downwelling(sw,ocean_albedo)
    Ice_sw  = pypwp.sw_downwelling(sw,si_albedo)
    Ice_lw  = pypwp.lw_downwelling(lw,si_emiss)
    ice_q=Ice_lw+Ice_sw
    ##  sensible and latent heat
    o_sens 	= pypwp.ao_sens(T_so,T_a,U_a,cd_ocean)
    sat_sp_hum 	= pypwp.saturation_spec_hum(T_so)
    o_lat 	= pypwp.ao_latent(T_so,U_a,sat_sp_hum,sp_hum,cd_ocean)

    ##  group terms (surface v penetrating)
    q_out	= OLR + o_sens + o_lat-ILR
    q_in	= ISW

    ##  buoyancy budget of Open ocean
    evap 	= o_lat/(1000.*Latent_vapor)
    emp 	= evap-precip #precip from forcing

    # may not be necessary here as not changed from previous calc? VVV

    ml_depth=z[ml_index]
    temp[0:ml_index + 1] = np.mean(temp[0:ml_index + 1])
    salt[0:ml_index + 1] = np.mean(salt[0:ml_index + 1])

    #                                                              ^^^

    #######################################
    ##   CALCULATE FLUXES OF HEAT/SALT   ##
    #######################################

    ##  flux surface fluxes evenly across existing ML #ice here perfectly reflective
    temp[0:ml_index+1]+=((1.-A)*(q_in)+A*Ice_sw)*absrb[0:ml_index+1]*dt/(dz*density[0:ml_index+1]*cp_ocean)
    temp[0:ml_index + 1] = np.mean(temp[0:ml_index + 1])

    density[0:ml_index + 1] = pypwp.density_0(temp[0:ml_index + 1],salt[0:ml_index + 1])

    temp[0:ml_index+1] -= ((1.-A)*q_out) * (dt / (density[0] * cp_ocean * ml_depth))

    salt[0:ml_index+1] = salt[0:ml_index+1]/(1.-(1.-A)*emp*dt/ml_depth)


    # do Biddle-Clark ice scheme as in UEA thesis ch5 (2016) modified basal and added conductivity.
    if bc_ice == 1:
        t_fp = pypwp.liquidous(salt[0])
        if temp[0] < t_fp:
            mr = (density[0] * cp_ocean * (temp[0] - t_fp)) / 1000.0 / Latent_fusion
            temp[0:ml_index + 1] = t_fp
            salt[0:ml_index + 1] = salt[0:ml_index + 1] / (1 + A * mr)
            h_i -= mr * ml_depth
            A = A_grow  # matter if this in beginning or end of if?
        elif h_i > 0.:
            mr = (density[0] * cp_ocean * (temp[0] - t_fp)) / 1000.0 / Latent_fusion
            temp[0:ml_index + 1] = t_fp
            salt[0:ml_index + 1] = salt[0:ml_index + 1] / (1 + (A * mr))
            h_i = h_i - mr * ml_depth
            A = A_melt
        else:
            mr = 0.
            A = 0.
            h_i = 0.
        t_flux = 0#mr*1000.0*Latent_fusion
        sw_flux=salt[0]*A*mr

    elif kt_ice==1:
        u_star_l = np.sqrt(cd_ocean * rho_air_ref / rho_ocean_ref) * U_a
        u_star_i = np.sqrt(cd_ice * rho_air_ref / rho_ocean_ref) * U_a
        t_fp = pypwp.liquidous(salt[0])
        if h_i>0.:
            T_si = pypwp.findroot(temp[0] - 20.0, temp[0] + 20.0, t_fp, h_snow, h_i, T_a, U_a, lw, sw, sp_hum)
            cond_flux=pypwp.ice_cond_heat(T_si,t_fp,h_i,h_snow)
        else:
            cond_flux=0.
        basal = (density[0] * cp_ocean * u_star_i * Stanton * (temp[0] - t_fp))
        mr=(basal-cond_flux)/ rho_ice_ref / Latent_fusion
        if temp[0]<t_fp:
            if A<A_grow:
                latheat=(1.-A)*(density[0] * cp_ocean * u_star_l  * (temp[0] - t_fp))
                dA = latheat/(Latent_fusion * rho_ice_ref * h_i)
                r_base=0.
                A-=dA*dt
                ridge=0.0
            else:
                latheat=(1.-A)*(density[0] * cp_ocean * u_star_l  * (temp[0] - t_fp))
                dA=0.
                r_base=latheat
                A=A_grow
                ridge=latheat*(1.-A)/(Latent_fusion*rho_ice_ref)
        else:
            if A > A_melt:
                latheat = (1. - A) *(density[0] * cp_ocean * u_star_l *Stanton * (temp[0] - t_fp))
                dA =  (1. - R_b) * latheat / (Latent_fusion * rho_ice_ref * h_i)
                A += -dA * dt
                r_base=(latheat*R_b)
                mr += (r_base)/rho_ice_ref/Latent_fusion
                ridge=0.0

            else:
                latheat=0.
                dA=0.
                r_base=0.
                A=A_melt
                ridge=0.0

        h_i-=mr*dt

        if h_i<h_ice_min:
            h_i=h_ice_min
            mr=0.
            basal=0.

        sw_flux =(rho_ice_ref/rho_ocean_ref)*(salt[0]  - S_ice) * (A * (mr+ridge)+dA*h_i)
        t_flux = 1./(rho_ocean_ref*cp_ocean) * (A*basal+latheat)
        temp[0:ml_index + 1] += - t_flux* dt / ml_depth
        salt[0:ml_index + 1] += -sw_flux*dt/ml_depth

        if A>A_melt:
            A-=Div*A

    ##  Penetrating shortwave below ML depth
    temp[ml_index+1:] = temp[ml_index+1:]+(1.-A)*ISW*absrb[ml_index+1:]*dt/(dz*density[ml_index+1:]*cp_ocean)
    density = pypwp.density_0(temp, salt)

    ##  make sure column still statically stable
    temp, salt, u, v, density, ml_index = pypwp.remove_static_instabilities(ml_index, temp, salt, u, v, density)
    ml_depth = z[ml_index]
    #######################################################
    ##   Calculate fluxes and Kraus-Turner type mixing   ##
    #######################################################

    if kt_switch==1:
        fw_flux= -salt[0]*(((1.-A)*emp))+sw_flux
        temp_flux=(1.-A)*(q_out-q_in*0.45)/(rho_ocean_ref*cp_ocean)-t_flux
        u_star = U_a*(((rho_air_ref/rho_ocean_ref)*cd_ocean))**(1./2.)		#neglects ice shear - assume u_i=u_ocean (urel=0)
                                                                                    #alt such as petty assume urel=u_10 not sure if better or worse. (cd_i>cd_o so wind mixing ^ with A ^

        Pw = ((2.*m_kt)*np.e**(-ml_depth/dw)*u_star**3) 			# Power for mixing supplied by wind
        Bo = ((g*alpha)*(temp_flux)) - (g*beta*(fw_flux))	# buoyancy forcing
        Pb = (ml_depth/2.)*((1.+n_kt)*Bo-(1.-n_kt)*abs(Bo))	# Power for mixing supplied by buoyancy change?

        we = (Pw+Pb)/(ml_depth*(g*alpha*(temp[0]-temp[ml_index+1])-g*beta*(salt[0]-salt[ml_index+1])))



    ###########################################################
    ##   Calculate mixed layer deepening from this balance   ##
    ###########################################################

        if we >= 0.:
            ml_depth_test = ml_depth + we * dt          #check motion due to ek over time step
            while ml_depth_test > (ml_depth + (dz / 2.)): # if moves more than dz/2 increment and recalc balance
                ml_index = ml_index + 1
                ml_depth = z[ml_index]
                Pw = ((2.*m_kt)*np.e**(-ml_depth/dw)*u_star**3)   # Power for mixing supplied by wind
                Pb = (ml_depth/2.)*((1.+n_kt)*Bo-(1.-n_kt)*abs(Bo))  # Power for mixing supplied by buoyancy change?
                we = (Pw+Pb)/(ml_depth*(g*alpha*(temp[0]-temp[ml_index+1])-g*beta*(salt[0]-salt[ml_index+1])))
                ml_depth_test = ml_depth+we*dt
        else:
            ml_depth_test = (Pw /(-Bo)) # sometimes this gives huge value when switching and results in artifacts.
            if ml_depth_test<ml_depth:
                ml_depth=ml_depth_test



        if ml_depth < ml_min:
            ml_depth=ml_min
            ml_index = int(round(ml_depth / dz))
        elif ml_depth>ml_max:
            ml_depth=ml_max
            ml_index = int(round(ml_depth / dz))
        else:
            ml_index = int(round(ml_depth / dz))
            ml_depth = z[ml_index]

        temp, salt, u, v, density = pypwp.mix(temp, salt, u, v, density, ml_index)


    ##   End Krauss-Turner   ##

    ##   Start PWP u/v stuff does nothing if rb=rg=0   ##
    ##  Time step the momentum equation.

    ##  Rotate the current throughout the water column
    u,v = pypwp.rot(ang,u,v)

    ##  Apply the wind stress to the mixed layer as it now exists.

    u_prev = u[0]
    v_prev = v[0]
    du = (tx/(ml_depth*density[0]))*dt
    dv = (ty/(ml_depth*density[0]))*dt

    u[0:ml_index+1] = u[0:ml_index+1]+du
    v[0:ml_index+1] = v[0:ml_index+1]+dv

    ## I've just commented this out. uconn is not set in mpwp or pwp found online
    ## think this is antiquated section
#  Apply drag to the current (this is a horrible parameterization of
#  inertial-internal wave dispersion).

    #if ucon > 1E-10:
    #    u = u*(1-dt*ucon)
    #    v = v*(1-dt*ucon)

    ##  Rotate another half time step.

    u,v = pypwp.rot(ang,u,v)

    ##  Finished with the momentum equation for this time step.
    ##  Do the bulk Richardson number instability form of mixing (as in PWP).

    if rb > 0: # Switch for bulk richardson
        ml_index,density,u,v,temp,salt = pypwp.bulk_mix(ml_index,rb,density,u,v,temp,salt,z,nz)
        ml_depth = z[ml_index]

    if rg > 0: # Switch for gradient richardson
        temp,salt,density,u,v,oxy,ml_index = pypwp.grad_mix(dz,g,rg,nz,z,temp,salt,density,u,v,oxy,ml_index)		# to do
        ml_depth = z[ml_index]


    oxy=pypwp.Oxygen_change(temp, salt, oxy, density, U_a, ml_depth,ml_index, dt,A)

    if iteration%dt_save==0: #actually save data.
        perc = iteration / maxiter * 100.
        print("%.2f" % perc,ml_depth,h_i,A)
        f_iter=filename+str(iteration/dt_save)
        mld_save.append(ml_depth)
        h_i_save.append(h_i)
        we_save.append(we)
        pb_save.append(Pb)
        pw_save.append(Pw)
        Bo_save.append(Bo)
        mr_save.append(mr)
        A_save.append(A)
        tml.append(temp[0])
        sml.append(salt[0])
        np.savez(os.path.join(save_path,f_iter),temp=temp,salt=salt,density=density,oxy=oxy,u=u,v=v)

    iteration+=1

filename='scalars'
np.savez(os.path.join(save_path,filename),mld=mld_save,we=we_save,pb=pb_save,pw=pw_save,hi=h_i_save,bo=Bo_save,mr=mr_save,A=A_save,tml=tml,sml=sml)
end = tp.time()
print(end - start)



print(base_path, filename, "param_copy.py")







	






































