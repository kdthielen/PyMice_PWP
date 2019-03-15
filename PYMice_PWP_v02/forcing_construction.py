import numpy as np
import scipy.io as sio
import os

met_input_file = "AS_longmet.mat"
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


time = time_series.copy()
T_a_modified = np.zeros(len(time_series))
lw_modified = np.zeros(len(time_series))
sw_modified = np.zeros(len(time_series))
shum_modified = np.zeros(len(time_series))
precip_modified = np.zeros(len(time_series))
U_a_modified = np.zeros(len(time_series))
tx_modified = np.zeros(len(time_series))
ty_modified= np.zeros(len(time_series))

summer_index=T_a_series.argmax()
winter_index=T_a_series.argmin()
wind_max_index=U_a_series.argmax()
precip_max_index=precip_series.argmax()

T_a_modified[:]=T_a_series[summer_index]
lw_modified[:]= lw_series[summer_index]
sw_modified[:]=sw_series[summer_index]
shum_modified[:]=shum_series[summer_index]

np.savez("summer_nowind",time=time,tair=T_a_modified,lw=lw_modified,sw=sw_modified,shum=shum_modified,precip=precip_modified,U=U_a_modified,tx=tx_modified,ty=ty_modified)
U_a_modified[:] = U_a_series[wind_max_index]
tx_modified[:] = tx_series[wind_max_index]
ty_modified[:] = ty_series[wind_max_index]
np.savez("summer_wind",time=time,tair=T_a_modified,lw=lw_modified,sw=sw_modified,shum=shum_modified,precip=precip_modified,U=U_a_modified,tx=tx_modified,ty=ty_modified)

precip_modified[:]=precip_series[precip_max_index]

np.savez("summer_wind_precip",time=time,tair=T_a_modified,lw=lw_modified,sw=sw_modified,shum=shum_modified,precip=precip_modified,U=U_a_modified,tx=tx_modified,ty=ty_modified)



T_a_modified = np.zeros(len(time_series))
lw_modified = np.zeros(len(time_series))
sw_modified = np.zeros(len(time_series))
shum_modified = np.zeros(len(time_series))
precip_modified = np.zeros(len(time_series))
U_a_modified = np.zeros(len(time_series))
tx_modified = np.zeros(len(time_series))
ty_modified= np.zeros(len(time_series))









T_a_modified[:]=T_a_series[winter_index]
lw_modified[:]= lw_series[winter_index]
sw_modified[:]=sw_series[winter_index]
shum_modified[:]=shum_series[winter_index]


np.savez("winter_nowind",time=time,tair=T_a_modified,lw=lw_modified,sw=sw_modified,shum=shum_modified,precip=precip_modified,U=U_a_modified,tx=tx_modified,ty=ty_modified)

U_a_modified[:] = U_a_series[wind_max_index]
tx_modified[:] = tx_series[wind_max_index]
ty_modified[:] = ty_series[wind_max_index]
np.savez("winter_wind",time=time,tair=T_a_modified,lw=lw_modified,sw=sw_modified,shum=shum_modified,precip=precip_modified,U=U_a_modified,tx=tx_modified,ty=ty_modified)

precip_modified[:]=precip_series[precip_max_index]

np.savez("winter_wind_precip",time=time,tair=T_a_modified,lw=lw_modified,sw=sw_modified,shum=shum_modified,precip=precip_modified,U=U_a_modified,tx=tx_modified,ty=ty_modified)
