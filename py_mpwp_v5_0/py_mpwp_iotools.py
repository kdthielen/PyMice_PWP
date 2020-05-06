import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr

rho_ocean_ref	= 1026.		# ocean reference

##########################################
##   LOAD FORCING AND INITIAL PROFILE   ##
##########################################
def load_forcing_fields(forcing_file, time):

    if forcing_file[-3:]=="mat":  #this is chloes format and louise?

        import scipy.io as sio
        data = sio.loadmat(forcing_file)
        time_series = data['met']['time'][0,0][:,0]
        T_a_series = data['met']['tair'][0,0][:,0]
        lw_series = data['met']['lw'][0,0][:,0]
        sw_series = data['met']['sw'][0,0][:,0]
        shum_series = data['met']['shum'][0,0][:,0]
        precip_series = data['met']['precip'][0,0][:,0]
        U_a_series = data['met']['U'][0,0][:,0]
        tx_series = data['met']['tx'][0,0][:,0]
        ty_series = data['met']['ty'][0,0][:,0]
        
        # interpolate forcing data on to the simulation timegrid
        forcing = {'time'  : time,
                   'T_a'   : np.interp(time, time_series,T_a_series    ),
                   'lw'    : np.interp(time, time_series,lw_series     ),
                   'sw'    : np.interp(time, time_series,sw_series     ),
                   'shum'  : np.interp(time, time_series,shum_series   ),
                   'precip': np.interp(time, time_series,precip_series ),
                   'U_a'   : np.interp(time, time_series,U_a_series    ),
                   'tx'    : np.interp(time, time_series,tx_series     ),
                   'ty'    : np.interp(time, time_series,ty_series     )
                  }
        forcing =  pd.DataFrame(forcing)
        forcing.to_xarray().to_netcdf('output_forcing.nc',mode='w')


    # Example of an npz file as made from provided scripts.
    elif forcing_file[-3:]=="npz":
        #fname='forcing_test_60.npz'
        met_forcing=np.load(forcing_file)
        time_series = met_forcing['time']
        T_a_series = met_forcing['tair']
        lw_series = met_forcing['lw']
        sw_series = met_forcing['sw']
        shum_series = met_forcing['shum']
        precip_series = met_forcing['precip']
        u10_series = met_forcing['u10']
        v10_series = met_forcing['v10']
        U_a_series = np.sqrt(np.square(u10_series)+np.square(v10_series))
        def tau_from_u10(u10,rho_air_ref=1.275,cd_air=0.0015):
            #cd_air=(0.10+0.13*U_a_series-0.0022*U_a_series**2)*10**(-3)
            #txi=rho_air_ref*np.abs(u10_series)*u10_series*2.36*10**(-3)/3.
            #tyi=rho_air_ref*np.abs(v10_series)*v10_series*2.36*10**(-3)/3.
            return rho_air_ref*np.abs(u10)*u10*cd_air
        tx_series = tau_from_u10(u10_series)
        ty_series = tau_from_u10(v10_series)
        
        # interpolate forcing data on to the simulation timegrid
        forcing = {'time'  : time,
                   'T_a'   : np.interp(time, time_series,T_a_series    ),
                   'lw'    : np.interp(time, time_series,lw_series     ),
                   'sw'    : np.interp(time, time_series,sw_series     ),
                   'shum'  : np.interp(time, time_series,shum_series   ),
                   'precip': np.interp(time, time_series,precip_series ),
                   'U_a'   : np.interp(time, time_series,U_a_series    ),
                   'tx'    : np.interp(time, time_series,tx_series     ),
                   'ty'    : np.interp(time, time_series,ty_series     )
                  }
        forcing =  pd.DataFrame(forcing)
        forcing.to_xarray().to_netcdf('output_forcing.nc',mode='w')

    # read from output_forcing.nc (obtained using xarray 'to_netcdf' function)
    elif forcing_file[-3:]==".nc":
        forcing = xr.open_dataset(forcing_file).to_dataframe()
        

    
    return forcing


###########################################
##  -- Load initial t,s profile data. -- ## Here are a couple examples of filetypes being loaded in
###########################################
def load_init_state(init_state_file,z,log_file):
#  detect filetype -> load -> check depth -> interpolate (fill surface nans with shallowest datapoint)

    if init_state_file[-3:]=="npz":
       # profile_input_file = "/home/thielen/Desktop/ttest/soccom_prof.npz"
        initial_profile = np.load(init_state_file)
        initial_z       = initial_profile['depth']
        initial_temp    = initial_profile['temp']
        initial_salt    = initial_profile['salt']
        initial_oxy     = np.zeros(len(initial_temp))
        initial_u       = np.zeros(len(initial_temp))
        initial_v       = np.zeros(len(initial_temp))
        # convert oxygen into umol/kg
        oxy = oxy * 44.658
        oxy = oxy / rho_ocean_ref
        oxy = oxy * 1000

    elif init_state_file[-3:]=="mat" :
        import scipy.io as sio
        initial_profile = sio.loadmat(init_state_file)
        initial_z       = initial_profile['profile']['z'][0, 0][0, :]
        initial_temp    = initial_profile['profile']['t'][0, 0][0, :]
        initial_salt    = initial_profile['profile']['s'][0, 0][0, :]
        initial_oxy     = initial_profile['profile']['oxy'][0, 0][0, :] #if you have no oxy then just make a same size array of zeros
        initial_u       = np.zeros(len(initial_temp))
        initial_v       = np.zeros(len(initial_temp))
        # convert oxygen into umol/kg
        oxy = oxy * 44.658
        oxy = oxy / rho_ocean_ref
        oxy = oxy * 1000
        
    elif init_state_file[-3:]==".nc" :
        with nc.Dataset(init_state_file, 'r') as nc_fid:
            initial_z       = nc_fid['depth'][:]
            initial_temp    = nc_fid['theta'][:]
            initial_salt    = nc_fid['salt'][:]
            initial_oxy     = nc_fid['oxy'][:]
            initial_u       = nc_fid['uvel'][:]
            initial_v       = nc_fid['vvel'][:]

    else:
        print('No initial state loaded: start from default state', file=open(log_file, 'a'))
        initial_z       = z
        initial_temp    = np.zeros(z.shape) + 10.
        initial_salt    = np.zeros(z.shape) + 35.
        initial_oxy     = np.zeros(z.shape)
        initial_u       = np.zeros(z.shape)
        initial_v       = np.zeros(z.shape)


    # Check depth domain of initial profile and truncate to deepest point of observations if shorter
    max_depth = np.max(z)
    dz        = z[1] - z[0]
    if initial_z[-1] < max_depth :
        maxdepth = dz * initial_z[-1] // dz
        nz = int(maxdepth // dz) + 1
        z  = np.arange(0, nz) * dz
        print('Maximum depth: {} meters. Truncated because input profile was too short.'.format(max(z)), file=open(log_file, 'a'))
    else:
        print('Maximum depth: {} meters'.format(max(z)), file=open(log_file, 'a'))

    #  -- Interpolate the profile variables at dz resolution. --
    temp	= np.interp(z,initial_z,initial_temp)
    salt	= np.interp(z,initial_z,initial_salt)
    oxy     = np.interp(z,initial_z,initial_oxy )
    u       = np.interp(z,initial_z,initial_u   )
    v       = np.interp(z,initial_z,initial_v   )

    # save in a global variable
    return z, temp, salt, oxy, u, v


def init_output(time,depth,niter_save):

    nrecord      = len(time[::niter_save])
    empty_1D     = np.zeros(nrecord)
    empty_2D     = np.zeros((nrecord,len(depth)))
    output  = xr.Dataset(data_vars={
                        "iter"    :(["time"],empty_1D.copy()),
                        "ml_index":(["time"],empty_1D.copy()),
                        "ml_depth":(["time"],empty_1D.copy()),
                        "si_thick":(["time"],empty_1D.copy()),
                        "si_conc" :(["time"],empty_1D.copy()),
                        "sw_flux" :(["time"],empty_1D.copy()),
                        "t_flux"  :(["time"],empty_1D.copy()),
                        "u_star_l":(["time"],empty_1D.copy()),
                        "theta"   :(["time","depth"],empty_2D.copy()),
                        "salt"    :(["time","depth"],empty_2D.copy()),
                        "oxy"     :(["time","depth"],empty_2D.copy()),
                        "sigma0"  :(["time","depth"],empty_2D.copy()),
                        "uvel"    :(["time","depth"],empty_2D.copy()),
                        "vvel"    :(["time","depth"],empty_2D.copy())}, 
                coords={"depth": depth, 
                        "time":  time[::niter_save]})
    return output


def save_output(output,nsave,niter,ml_index,ml_depth,h_i,A,temp,salt,oxy,density,u,v,sw_flux,t_flux,u_star_l):
    
    output["iter"    ][nsave] = niter
    output["ml_index"][nsave] = ml_index
    output["ml_depth"][nsave] = ml_depth
    output["si_thick"][nsave] = h_i
    output["si_conc" ][nsave] = A
    output["sw_flux" ][nsave] = sw_flux
    output["t_flux"  ][nsave] = t_flux
    output["u_star_l"][nsave] = u_star_l
    output["theta"   ][nsave] = temp
    output["salt"    ][nsave] = salt
    output["oxy"     ][nsave] = oxy
    output["sigma0"  ][nsave] = density
    output["uvel"    ][nsave] = u
    output["vvel"    ][nsave] = v
    
    return output


