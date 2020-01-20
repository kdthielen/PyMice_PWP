
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import os
import pypwp_functions as pypwp
from collections import OrderedDict
from scipy.interpolate import griddata

base_paths=['year_rb_0.5/','year_rb_0.6/','year_rb_0.7/','year_ref_/','year_rb_0.9/']
base_paths=['year_advect_0.2/','year_advect_0.3/','year_advect_0.4/','year_ref_/','year_advect_0.6/','year_advect_0.7/','year_advect_0.8/']
base_paths=['year_hi0_0.1/','year_hi0_0.2/','year_hi0_0.3/','year_hi0_0.4/','year_ref_/','year_hi0_0.6/','year_hi0_0.7/','year_hi0_0.8/','year_hi0_0.9/','year_hi0_1.0/']
base_paths=['year_A0_0.5/','year_A0_0.6/','year_A0_0.7/','year_A0_0.8/','year_ref_/','year_A0_0.95/']
base_paths=['year_cda_0.001/','year_cda_0.00125/','year_ref_/','year_cda_0.00175/','year_cda_0.002/']
base_paths=['year_ekman_0.0/','year_ekman_0.0000012/','year_ref_/','year_ekman_0.0000016/','year_ekman_0.0000018/','year_ekman_0.000002/']
base_paths=['year_hs_0.0/','year_hs_0.05/','year_ref_/','year_hs_0.15/','year_hs_0.2/','year_hs_0.25/','year_hs_0.3/']
#base_paths=['final_ekman_0.0/','final_ref/']
#base_paths=['final_advectchange/','final_ref/']
savename='year_hs'


a336_profile_input_file = '336_prof.npz'
a336_profile=np.load(a336_profile_input_file)
a336_z=a336_profile['depth']
a336_salt=a336_profile['salt']
a336_temp=a336_profile['temp']


a366_profile_input_file = "368_prof.npz"
a366_profile=np.load(a366_profile_input_file)
a366_z=a366_profile['depth']
a366_salt=a366_profile['salt']
a366_temp=a366_profile['temp']


a396_profile_input_file = "400_prof.npz"
a396_profile=np.load(a396_profile_input_file)
a396_z=a396_profile['depth']
a396_salt=a396_profile['salt']
a396_temp=a396_profile['temp']


a426_profile_input_file = "424_prof.npz"
a426_profile=np.load(a426_profile_input_file)
a426_z=a426_profile['depth']
a426_salt=a426_profile['salt']
a426_temp=a426_profile['temp']


a336_density=pypwp.density_0(a336_temp,a336_salt)
a366_density=pypwp.density_0(a366_temp,a366_salt)
a396_density=pypwp.density_0(a396_temp,a396_salt)
a426_density=pypwp.density_0(a426_temp,a426_salt)




cp=4190.




filenum=[0,81,164,226]  # 84 for 60 148 for 57
mli=251   
dz=3.0
z = np.arange(0,mli ) * dz

gridded_336_temp=griddata(a336_z,a336_temp,z)
gridded_336_salt=griddata(a336_z,a336_salt,z)

gridded_366_temp=griddata(a366_z,a366_temp,z)
gridded_366_salt=griddata(a366_z,a366_salt,z)

gridded_396_temp=griddata(a396_z,a396_temp,z)
gridded_396_salt=griddata(a396_z,a396_salt,z)

gridded_426_temp=griddata(a426_z,a426_temp,z)
gridded_426_salt=griddata(a426_z,a426_salt,z)


gridded_336_salt[np.isnan(gridded_336_salt)]=a336_salt[0]
gridded_366_salt[np.isnan(gridded_366_salt)]=a366_salt[0]
gridded_396_salt[np.isnan(gridded_396_salt)]=a396_salt[0]
gridded_426_salt[np.isnan(gridded_426_salt)]=a426_salt[0]

gridded_336_temp[np.isnan(gridded_336_temp)]=a336_temp[0]
gridded_366_temp[np.isnan(gridded_366_temp)]=a366_temp[0]
gridded_396_temp[np.isnan(gridded_396_temp)]=a396_temp[0]
gridded_426_temp[np.isnan(gridded_426_temp)]=a426_temp[0]

gridded_336_density=pypwp.density_0(gridded_336_temp,gridded_336_salt)
gridded_366_density=pypwp.density_0(gridded_366_temp,gridded_366_salt)
gridded_396_density=pypwp.density_0(gridded_396_temp,gridded_396_salt)
gridded_426_density=pypwp.density_0(gridded_426_temp,gridded_426_salt)


data_path=[]
save_path=[]
comp_plot_output="comparisons/"
if not os.path.exists(comp_plot_output):
    os.makedirs(comp_plot_output)
if not os.path.exists(comp_plot_output+savename):
    os.makedirs(comp_plot_output+savename)
i=0

# construct data and save paths assuming data saved in base_path/data/...
# create folder for plots
for fname in base_paths:
    data_path.append(fname+"data/")
    save_path.append(fname+"plots")  # tell it where to save the plots to be made
    if not os.path.exists(save_path[i]):
        os.makedirs(save_path[i])
    i+=1

## initialize arrays


temp=np.zeros((mli,filenum[-1]+1,len(base_paths)))
salt=np.zeros((mli,filenum[-1]+1,len(base_paths)))
density=np.zeros((mli,filenum[-1]+1,len(base_paths)))

z0=np.arange(mli)*3.
z1=np.arange(mli)*3.


# load data into arrays and optionally plot individual profiles


for j in range(len(base_paths)):
    for i in range(0, filenum[-1] + 1):
        f_name = 'profiles_' + str(i) + '.0.npz'
        file_temp = np.load(os.path.join(data_path[j], f_name))
        temp_temp=file_temp['temp']
        salt_temp = file_temp['salt']
        density_temp = file_temp['density']
        temp[:,i,j]=temp_temp
        salt[:,i,j]=salt_temp
        density[:,i,j]=density_temp
       
fig = plt.figure()

ax1 = fig.add_subplot(141)
plt.plot(a336_temp, -1 * a336_z,ls='--',color='k', label='Obs_Start ')
plt.xlabel('Temperature C')
plt.ylabel('Depth (m)')
plt.xlim(-3, 3)
plt.ylim(-300,0)

ax2 = fig.add_subplot(142)
plt.plot(a366_temp, -1 * a366_z,ls='--',color='k', label='Obs_Oct ')
plt.xlabel('Temperature C')
plt.xlim(-3, 3)
plt.ylim(-300,0)
plt.setp(ax2.get_yticklabels(), visible=False)

ax3 = fig.add_subplot(143)
plt.plot(a396_temp, -1 * a396_z,ls='--',color='k', label='Obs_Dec ')
plt.xlabel('Temperature C')
plt.xlim(-3, 3)
plt.ylim(-300,0)
plt.setp(ax3.get_yticklabels(), visible=False)

ax4 = fig.add_subplot(144)
plt.plot(a426_temp, -1 * a426_z,ls='--',color='k', label='Obs_March ')
plt.xlabel('Temperature C')
plt.xlim(-3, 3)
plt.ylim(-300,0)
plt.setp(ax4.get_yticklabels(), visible=False)

for j in range(len(base_paths)):    
    temp_temp336 = temp[:,filenum[0],j]
    temp_temp366 = temp[:,filenum[1],j]
    temp_temp396 = temp[:,filenum[2],j]
    temp_temp426 = temp[:,filenum[3],j]
    ax1 = fig.add_subplot(141)
    plt.plot(temp_temp336, -1 * z, label=base_paths[j])
    

    ax2 = fig.add_subplot(142)
    plt.plot(temp_temp366, -1 * z, label=base_paths[j])
    
    
    ax3 = fig.add_subplot(143)
    plt.plot(temp_temp396, -1 * z, label=base_paths[j])
    

    ax4 = fig.add_subplot(144)
    plt.plot(temp_temp426, -1 * z, label=base_paths[j])
    

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(),loc='upper center',bbox_to_anchor=(0.5, 0.175),ncol=3)
plt.gcf().subplots_adjust(bottom=0.3)

#plt.tight_layout()
#plt.show()
plt.savefig(str(comp_plot_output)+str(savename)+'/temp_compare.png')   #todo make this a specified folder
plt.close()
plt.clf()


fig = plt.figure()

ax1 = fig.add_subplot(141)
plt.plot(a336_salt, -1 * a336_z,ls='--',color='k', label='Obs_Start ')
plt.ylim(-300,0)
plt.ylabel('Depth (m)')
plt.xlim(33.4, 34.8)
plt.xlabel('Salinity (psu)')

ax2 = fig.add_subplot(142)
plt.plot(a366_salt, -1 * a366_z,ls='--',color='k', label='Obs_Oct ')
plt.ylim(-300,0)
plt.ylabel('Depth (m)')
plt.xlim(33.4, 34.8)
plt.xlabel('salinity')
plt.setp(ax2.get_yticklabels(), visible=False)

ax3 = fig.add_subplot(143)
plt.plot(a396_salt, -1 * a396_z,ls='--',color='k', label='Obs_Dec ')
plt.ylim(-300,0)
plt.ylabel('Depth (m)')
plt.xlim(33.4, 34.8)
plt.xlabel('Salinity (psu)')
plt.setp(ax3.get_yticklabels(), visible=False)

ax4 = fig.add_subplot(144)
plt.plot(a426_salt, -1 * a426_z,ls='--',color='k', label='Obs_March ')
plt.ylim(-300,0)
plt.ylabel('Depth (m)')
plt.xlim(33.4, 34.8)
plt.xlabel('Salinity (psu)')
plt.setp(ax4.get_yticklabels(), visible=False)


j=0

for j in range(len(base_paths)):    
    temp_salt336 = salt[:,filenum[0],j]
    temp_salt366 = salt[:,filenum[1],j]
    temp_salt396 = salt[:,filenum[2],j]
    temp_salt426 = salt[:,filenum[3],j]

    ax1a = fig.add_subplot(141)
    plt.plot(temp_salt336, -1 * z, label=base_paths[j])
    plt.ylim(-300,0)
    plt.ylabel('Depth (m)')
    plt.xlim(33.4, 34.8)
    plt.xlabel('Salinity (psu)')

    ax2a = fig.add_subplot(142)
    plt.plot(temp_salt366, -1 * z, label=base_paths[j])
    plt.ylim(-300,0)

    plt.xlim(33.4, 34.8)
    plt.xlabel('Salinity (psu)')
    plt.setp(ax2a.get_yticklabels(), visible=False)

    ax3a = fig.add_subplot(143)
    plt.plot(temp_salt396, -1 * z, label=base_paths[j])
    plt.ylim(-300,0)

    plt.xlim(33.4, 34.8)
    plt.xlabel('Salinity (psu)')
    plt.setp(ax3a.get_yticklabels(), visible=False)

    ax4a = fig.add_subplot(144)
    plt.plot(temp_salt426, -1 * z, label=base_paths[j])
    plt.ylim(-300,0)

    plt.xlim(33.4, 34.8)
    plt.xlabel('Salinity (psu)')
    plt.setp(ax4a.get_yticklabels(), visible=False)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(),loc='upper center',bbox_to_anchor=(0.5, 0.175),ncol=3)
plt.gcf().subplots_adjust(bottom=0.3)

#plt.tight_layout()
#plt.show()
plt.savefig(str(comp_plot_output)+str(savename)+'/salt_compare.png')   #todo make this a specified folder
plt.close()
plt.clf()

fig = plt.figure()

ax1 = fig.add_subplot(141)
plt.plot(a336_density, -1 * a336_z,ls='--',color='k', label='Obs_Start ')
plt.ylim(-300,0)
plt.xlabel('Density (kg/m^3)')
plt.ylabel('Depth (m)')
plt.xlim(1027.0,1027.9)
plt.xticks([1027,1027.2,1027.4,1027.6,1027.8])

ax2 = fig.add_subplot(142)
plt.plot(a366_density, -1 * a366_z,ls='--',color='k', label='Obs_Oct ')
plt.ylim(-300,0)
plt.xlabel('Density (kg/m^3)')
plt.xlim(1027.0,1027.9)
plt.xticks([1027,1027.2,1027.4,1027.6,1027.8])
plt.setp(ax2.get_yticklabels(), visible=False)

ax3 = fig.add_subplot(143)
plt.plot(a396_density, -1 * a396_z,ls='--',color='k', label='Obs_Dec ')
plt.ylim(-300,0)
plt.xlabel('Density (kg/m^3)')
plt.xlim(1027.0,1027.9)
plt.xticks([1027,1027.2,1027.4,1027.6,1027.8])
plt.setp(ax3.get_yticklabels(), visible=False)

ax4 = fig.add_subplot(144)
plt.plot(a426_density, -1 * a426_z,ls='--',color='k', label='Obs_March ')
plt.ylim(-300,0)
plt.xlabel('Density (kg/m^3)')
plt.xlim(1027.0,1027.9)
plt.xticks([1027,1027.2,1027.4,1027.6,1027.8])
plt.setp(ax4.get_yticklabels(), visible=False)

j=0
for j in range(len(base_paths)):    
    temp_dens336 = density[:,filenum[0],j]
    temp_dens366 = density[:,filenum[1],j]
    temp_dens396 = density[:,filenum[2],j]
    temp_dens426 = density[:,filenum[3],j]
    
    
    ax1 = fig.add_subplot(141)
    plt.plot(temp_dens336, -1 * z, label=base_paths[j])

    ax2 = fig.add_subplot(142)
    plt.plot(temp_dens366, -1 * z, label=base_paths[j])
    
    ax3 = fig.add_subplot(143)
    plt.plot(temp_dens396, -1 * z, label=base_paths[j])
    
    ax4 = fig.add_subplot(144)
    plt.plot(temp_dens426, -1 * z, label=base_paths[j])

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(),loc='upper center',bbox_to_anchor=(0.5, 0.175),ncol=3)
plt.gcf().subplots_adjust(bottom=0.3)

#plt.tight_layout()
#plt.show()
plt.savefig(str(comp_plot_output)+str(savename)+'/density_compare.png')   #todo make this a specified folder


plt.close()
plt.clf()
finind=149
for j in range(len(base_paths)): 
	temp_salt336 = salt[:,filenum[0],j]
	temp_temp336 = temp[:,filenum[0],j]
	
	plt.title('July')
	plt.scatter(temp_salt336[0:101],temp_temp336[0:101],c=z[0:101])
	plt.plot(temp_salt336[0:101],temp_temp336[0:101],ls=':',label=base_paths[j])
	plt.xlabel('Salt')
	plt.xlim(33.8,34.8)
	plt.ylabel('Temp')
	plt.ylim(-2.0,2.0)


plt.scatter(a336_salt[0:finind],a336_temp[0:finind],c=a336_z[0:finind],marker='v')
plt.plot(a336_salt[0:finind],a336_temp[0:finind],ls='-',color='k',lw=2,label='obs')
plt.colorbar()
plt.legend()
plt.savefig(str(comp_plot_output)+str(savename)+'/'+'July_T_S_SIM.png')   #todo make this a
plt.clf()

for j in range(len(base_paths)): 
	temp_salt366 = salt[:,filenum[1],j]
	temp_temp366 = temp[:,filenum[1],j]

	plt.title('Oct')
	plt.scatter(temp_salt366[0:101],temp_temp366[0:101],c=z[0:101])
	plt.plot(temp_salt366[0:101],temp_temp366[0:101],ls=':',label=base_paths[j])
	plt.xlabel('Salt')
	plt.xlim(33.8,34.8)
	plt.ylabel('Temp')
	plt.ylim(-2.0,2.0)


plt.scatter(a366_salt[0:finind],a366_temp[0:finind],c=a366_z[0:finind],marker='v')
plt.plot(a366_salt[0:finind],a366_temp[0:finind],ls='-',color='k',lw=2,label='obs')
plt.colorbar()
plt.legend()
plt.savefig(str(comp_plot_output)+str(savename)+'/'+'OCT_T_S_SIM.png') 
plt.clf()

for j in range(len(base_paths)): 
	temp_salt396 = salt[:,filenum[2],j]
	temp_temp396 = temp[:,filenum[2],j]

	plt.title('Dec')
	plt.scatter(temp_salt396[0:101],temp_temp396[0:101],c=z[0:101])
	plt.plot(temp_salt396[0:101],temp_temp396[0:101],ls=':',label=base_paths[j])
	plt.xlabel('Salt')
	plt.xlim(33.8,34.8)
	plt.ylabel('Temp')
	plt.ylim(-2.0,2.0)


plt.scatter(a396_salt[0:finind],a396_temp[0:finind],c=a396_z[0:finind],marker='v')
plt.plot(a396_salt[0:finind],a396_temp[0:finind],ls='-',color='k',lw=2,label='obs')
plt.colorbar()
plt.legend()
plt.savefig(str(comp_plot_output)+str(savename)+'/'+'Dec_T_S_SIM.png') 
plt.clf()

for j in range(len(base_paths)): 
	temp_temp426 = temp[:,filenum[3],j]
	temp_salt426 = salt[:,filenum[3],j]

	plt.title('March')
	plt.scatter(temp_salt426[0:101],temp_temp426[0:101],c=z[0:101])
	plt.plot(temp_salt426[0:101],temp_temp426[0:101],ls=':',label=base_paths[j])
	plt.xlabel('Salt')
	plt.xlim(33.8,34.8)
	plt.ylabel('Temp')
	plt.ylim(-2.0,2.0)


plt.scatter(a426_salt[0:finind],a426_temp[0:finind],c=a426_z[0:finind],marker='v')
plt.plot(a426_salt[0:finind],a426_temp[0:finind],ls='-',color='k',lw=2,label='obs')
plt.colorbar()
plt.legend()
plt.savefig(str(comp_plot_output)+str(savename)+'/'+'March_T_S_SIM.png') 
plt.clf()
#####################################
"""
for j in range(3,4):#len(base_paths[0])): 
	colormap=plt.get_cmap('viridis')
	cnorm=colors.Normalize(vmin=filenum[2],vmax=filenum[3])
	scalarMap=cmx.ScalarMappable(norm=cnorm,cmap=colormap)
	values=np.arange(filenum[2],filenum[3]+1)
	for i in range(filenum[2],filenum[3]+1):
		temp_temp426 = temp[:,i,j]
		temp_salt426 = salt[:,i,j]
		colorVal = scalarMap.to_rgba(i)
		plt.title('March')
		#plt.scatter(temp_salt426[0:101],temp_temp426[0:101],c=z[0:101])
		plt.plot(temp_salt426[0:101],temp_temp426[0:101],ls='-',label=i,color=colorVal)
		plt.xlabel('Salt')
		plt.xlim(33.8,34.8)
		plt.ylabel('Temp')
		plt.ylim(-2.0,2.0)
"""
###############################################

Q_336=np.sum(gridded_336_density[0:100]*cp*gridded_336_temp[0:100]*dz)
Q_366=np.sum(gridded_366_density[0:100]*cp*gridded_366_temp[0:100]*dz)
Q_396=np.sum(gridded_396_density[0:100]*cp*gridded_396_temp[0:100]*dz)
Q_426=np.sum(gridded_426_density[0:100]*cp*gridded_426_temp[0:100]*dz)


Q_temp336=np.sum(temp_dens336[0:100]*cp*temp_temp336[0:100]*dz)
Q_temp366=np.sum(temp_dens366[0:100]*cp*temp_temp366[0:100]*dz)
Q_temp396=np.sum(temp_dens396[0:100]*cp*temp_temp396[0:100]*dz)
Q_temp426=np.sum(temp_dens426[0:100]*cp*temp_temp426[0:100]*dz)
print(z[100])
print(Q_336,Q_366,Q_396,Q_426)
print(Q_temp336,Q_temp366,Q_temp396,Q_temp426)
 
Q=np.zeros((251,filenum[-1]+1))
Q50=np.zeros(filenum[-1]+1)
Qml=np.zeros(filenum[-1]+1)
scalars=np.load(os.path.join(data_path[0],'scalars.npz'))
mld=scalars['mld']
ib_save=scalars['ice_basal']
osens_save=scalars['o_sens']
o_lat_save=scalars['o_lat']
olr_save=scalars['olr']
isw_save=scalars['sw']
A=scalars['A']
time = scalars['time']

print mld
for i in range (0,filenum[-1]+1):
    density_q= density[:,i,0]
    temperature=temp[:,i,0]
    mli=int(mld[i]//3)
    Q[:,i]=density_q[:]*cp*temperature[:]*dz
    plt.plot(Q[0:100,i],-z[0:100])
    plt.xlim(-3e7,3e7)
    plt.xlabel(r'$\rho(z) c_p T(z)\Delta z$')
    plt.ylabel('Depth (m)')
    plt.title('Heat Evolution of '+base_paths[0])
    plt.savefig(str(comp_plot_output)+str(savename)+'/q_profs/q_test_'+str(i).zfill(3)+'.png')
    plt.clf()
	#Q50[i]=np.sum(density_q[0:18]*cp*temperature[0:18]*dz)
	#Qml[i]=np.sum(density_q[0:mli+1]*cp*temperature[0:mli+1]*dz)
plt.title(base_paths[0])
plt.plot(time,ib_save*A,label='basal')
plt.plot(time,-osens_save,label='sens')
plt.plot(time,-o_lat_save,label='lat')
plt.plot(time,-olr_save,label='olr')
plt.plot(time,isw_save,label='sw')
plt.legend()


#plt.plot(t,-mld*1e7)
#plt.show()
#plt.savefig(str(comp_plot_output)+str(savename)+'/Q_components.png')



