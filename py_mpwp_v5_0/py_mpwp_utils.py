import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr



###############################################
##       plot section of state               ##
###############################################
def plot_TS_section(fname='output.nc',name_fig='section.png'):

    with nc.Dataset(fname, 'r') as nc_fid:
        time       = nc_fid['time'][:]
        depth      = nc_fid['depth'][:]
        temp       = nc_fid['theta'][:,:]
        salt       = nc_fid['salt'][:,:]
        ml_depth   = nc_fid['ml_depth'][:]
        ml_index   = nc_fid['ml_index'][:]
        si_thick   = nc_fid['si_thick'][:]
        si_conc    = nc_fid['si_conc'][:]
    
    X, Y =np.meshgrid(time,depth)
    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 9), sharex=True)
    ax[0,0].plot(time,si_thick)
    ax[0,0].set_title('sea-ice thickness')

    ax[0,1].plot(time,si_conc*100)
    ax[0,1].set_title('sea-ice concentration')

    ax[1,1].plot(time,temp[:,0])
    ax[1,1].set_title('sea surface temperature')

    ax[2,1].plot(time,salt[:,0])
    ax[2,1].set_title('sea surface salinity')

    ax[1,0].pcolormesh(X, Y, temp.T)
    ax[1,0].set_ylim([-10,200])
    ax[1,0].invert_yaxis()
    ax[1,0].plot(time,ml_depth,'k')
    ax[1,0].set_title('temperature')
    
    ax[2,0].pcolormesh(X, Y, salt.T)
    ax[2,0].set_ylim([-10,200])
    ax[2,0].invert_yaxis()
    ax[2,0].plot(time,ml_depth,'k')
    ax[2,0].set_title('salinity')
    ax[2,0].set_xlabel('time [days]')

    plt.savefig(name_fig)



###############################################
##       plot state at a given time          ##
###############################################
def plot_state(fname='output.nc',name_fig='profile_TS.png',niter=-1):

    with nc.Dataset(fname, 'r') as nc_fid:
        time       = nc_fid['time'][niter]
        depth      = nc_fid['depth'][:]
        temp       = nc_fid['theta'][niter,:]
        salt       = nc_fid['salt'][niter,:]
        oxy        = nc_fid['oxy'][niter,:]
        uvel       = nc_fid['uvel'][niter,:]
        vvel       = nc_fid['vvel'][niter,:]
        ml_depth   = nc_fid['ml_depth'][niter]
        ml_index   = int(nc_fid['ml_index'][niter])
        si_thick   = nc_fid['si_thick'][niter]
        si_conc    = nc_fid['si_conc'][niter]

    min_temp   = np.min(temp)
    max_temp   = np.max(temp)
    min_salt   = np.min(salt)
    max_salt   = np.max(salt)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 9), sharey=True)
    plt.gca().set_ylim([-40,max(depth)+10])
    plt.gca().invert_yaxis()
    ax1.plot(temp,depth)
    ax1.set_title('temperature')
    ax1.plot(temp[ml_index],depth[ml_index],'ro')
    ax1.add_patch(patches.Rectangle((min_temp, -si_thick*10), max_temp-min_temp, si_thick*10))
    ax1.annotate('sea ice: {:5.2f}m, {:4.1f}%'.format(si_thick,si_conc*100), 
                 xy=(min_temp, -20),  
                 xycoords='data')

    ax2.plot(salt,depth)
    ax2.set_title('salinity')
    ax2.plot(salt[ml_index],depth[ml_index],'ro')
    ax2.add_patch(
        patches.Rectangle((min_salt, -si_thick*10), max_salt-min_salt, si_thick*10))

    plt.savefig(name_fig)


###############################################
##       plot state at a given time          ##
###############################################
def extract_state(fname='output.nc',name_state_file='state.nc',niter=0):

    with nc.Dataset(fname, 'r') as nc_fid:
        time       = nc_fid['time'][niter]
        depth      = nc_fid['depth'][:]
        temp       = nc_fid['theta'][niter,:]
        salt       = nc_fid['salt'][niter,:]
        oxy        = nc_fid['oxy'][niter,:]
        uvel       = nc_fid['uvel'][niter,:]
        vvel       = nc_fid['vvel'][niter,:]
        ml_depth   = nc_fid['ml_depth'][niter]
        ml_index   = int(nc_fid['ml_index'][niter])
        si_thick   = nc_fid['si_thick'][niter]
        si_conc    = nc_fid['si_conc'][niter]
    
    state = xr.Dataset(data_vars={
                        "time"    : time, 
                        "iter"    : niter, 
                        "theta"   :(["depth"],temp),
                        "salt"    :(["depth"],salt),
                        "oxy"     :(["depth"],oxy),
                        "uvel"    :(["depth"],uvel),
                        "vvel"    :(["depth"],vvel),
                        "ml_index": ml_index,
                        "ml_depth": ml_depth,
                        "si_thick": si_thick,
                        "si_conc" : si_conc},
                coords={"depth": depth})
    if fname[-3:]!='.nc':
        fname = fname+'.nc'
    state.to_netcdf(name_state_file,mode='w')

    
###############################################
##  diagnostic plots (to add to)/modularize  ##
###############################################
def plot_forcings(fname='output_forcing.nc',name_fig='forcing.png'):

    with nc.Dataset(fname, 'r') as nc_fid:
        time       = nc_fid['time'][:]
        T_a        = nc_fid['T_a'][:]
        lw         = nc_fid['lw'][:]
        sw         = nc_fid['sw'][:]
        shum       = nc_fid['shum'][:]
        precip     = nc_fid['precip'][:]
        U_a        = nc_fid['U_a'][:]
        tx         = nc_fid['tx'][:]
        ty         = nc_fid['ty'][:]

    # print(OLR_ice, ILR_ice, ISW_ice, i_sens, i_lat, T_si)
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(411)
    plt.title('' )
    plt.plot(time, lw, label='new')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel('lw')

    ax2 = fig.add_subplot(412)
    plt.plot(time, sw, label='new')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylabel('sw')

    ax3 = fig.add_subplot(413)
    plt.plot(time, T_a, label='new')
    plt.xlabel('time')
    plt.ylabel('T_{air}')

    ax4 = fig.add_subplot(414)
    plt.plot(time, U_a, label='new')
    plt.xlabel('time')
    plt.ylabel('U_{air}')

    plt.savefig(name_fig)

