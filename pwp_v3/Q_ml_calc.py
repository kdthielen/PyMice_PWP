import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import os
import pypwp_functions as pypwp
from collections import OrderedDict
from scipy.interpolate import griddata


filenum=226  # 84 for 60 148 for 57
mli=251     #196 for 60 201 for 57
dz=3.0
z = np.arange(0,mli ) * dz

base_paths=['final_rb_0.5/','final_rb_0.6/','final_rb_0.7/','final_ref/','final_rb_0.9/']
#base_paths=['final_advect_0.2/','final_advect_0.3/','final_advect_0.4/','final_ref/','final_advect_0.6/','final_advect_0.7/','final_advect_0.8/']
#base_paths=['final_hi0_0.1/','final_hi0_0.2/','final_hi0_0.3/','final_hi0_0.4/','final_ref/','final_hi0_0.6/','final_hi0_0.7/','final_hi0_0.8/','final_hi0_0.9/','final_hi0_1.0/']
#base_paths=['final_A0_0.5/','final_A0_0.6/','final_A0_0.7/','final_A0_0.8/','final_ref/','final_A0_0.95/']
#base_paths=['final_cda_0.001/','final_cda_0.00125/','final_ref/','final_cda_0.00175/','final_cda_0.002/']
#base_paths=['final_ekman_0.0/','final_ekman_0.0000012/','final_ref/','final_ekman_0.0000016/','final_ekman_0.0000018/','final_ekman_0.000002/']
#base_paths=['final_hs_0.0/','final_hs_0.05/','final_ref/','final_hs_0.15/','final_hs_0.2/','final_hs_0.25/','final_hs_0.3/']

#base_paths=['final_ref/']
savename='rb'


data_path=[]
save_path=[]

comp_plot_output="comparisons/"
if not os.path.exists(comp_plot_output):
    os.makedirs(comp_plot_output)
if not os.path.exists(comp_plot_output+savename):
    os.makedirs(comp_plot_output+savename)




# construct data and save paths assuming data saved in base_path/data/...
# create folder for plots
i=0
for fname in base_paths:
    data_path.append(fname+"data/")
    save_path.append(fname+"plots")  # tell it where to save the plots to be made
    if not os.path.exists(save_path[i]):
        os.makedirs(save_path[i])
    i+=1

## initialize arrays

final_temp=np.zeros((mli,len(base_paths)))
final_salt=np.zeros((mli,len(base_paths)))
final_density=np.zeros((mli,len(base_paths)))
temp=np.zeros((mli,filenum+1,len(base_paths)))
salt=np.zeros((mli,filenum+1,len(base_paths)))
density=np.zeros((mli,filenum+1,len(base_paths)))

z0=np.arange(mli)*3.
z1=np.arange(mli)*3.
Q_ml=np.zeros(filenum+1)




cp_ocean	    = 4190.
#rho_0=np.zeros((mli,filenum+1,len(base_paths)))
for j in range(len(base_paths)):
	ml_depth=np.zeros(filenum+1,int)
	scalars=np.load(os.path.join(data_path[j],'scalars.npz'))
	for i in range(0, filenum + 1):
		f_name = 'profiles_' + str(i) + '.0.npz'
		file_temp = np.load(os.path.join(data_path[j], f_name))
		temp_temp=file_temp['temp']
		salt_temp = file_temp['salt']
		density_temp = file_temp['density']
		temp[:,i,j]=temp_temp
		salt[:,i,j]=salt_temp
		density[:,i,j]=density_temp
        
		check=0
		k=0
		while check==0 and i<250:
			k+=1
			#print(i,j)
			crit=-(density[0,i,j]-density[k,i,j])
			#print density[0,j,k], density[i,j,k], i, j
			if crit>0.03:
	       			check=k
				ml_depth[i]=int(check*3)
				
		Q_ml[i]=np.sum(density_temp[0:ml_depth[i]]*cp_ocean*(temp_temp[0:ml_depth[i]]-0))*dz/float(ml_depth[i])
	time=scalars['time']
	time=time-time[0]
	plt.plot(time,Q_ml,label=base_paths[j])
	

plt.xlabel('Days')
plt.ylabel("$Q_{ML}$")



plt.axhline(0, color='k')
plt.axvline(0, color='k')

plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig(str(comp_plot_output)+str(savename)+'/N_squared.png')
plt.clf()
