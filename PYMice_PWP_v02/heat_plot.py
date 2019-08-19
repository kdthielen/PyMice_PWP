import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import os
import pypwp_functions as pypwp
from collections import OrderedDict
from scipy.interpolate import griddata

def plot_tsd(t0, s0, d0, iteration):
    plt.clf()
    # print(OLR_ice, ILR_ice, ISW_ice, i_sens, i_lat, T_si)
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    plt.title('d,s,t at %.2f days' % iteration)
    plt.plot(d0, -z0, label='py')
    plt.plot(initial_density, -1 * initial_z, ls='--', color='k', label='obs start')
    plt.plot(final_density_0, -1 * final_z_0, color='k', label='obs end')
    plt.xlim(1027.4, 1027.8)
    plt.ylim(-300, 0)
    plt.xlabel('density kg/m^3')
    plt.ylabel('depth')
    plt.xlim(1027.2, 1027.9)
    plt.xticks([1027.2, 1027.4, 1027.6, 1027.8])
    ax2 = fig.add_subplot(132)
    plt.plot(s0, -z0, label='py')
    plt.plot(initial_salt, -1 * initial_z, ls='--', color='k', label='obs start')
    plt.plot(final_salt_0, -1 * final_z_0, color='k', label='obs end')
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.xlim(33.8, 34.8)
    plt.ylim(-300, 0)
    plt.xlabel('salinity')
    ax3 = fig.add_subplot(133)
    plt.plot(t0, -z0, label='py')
    plt.plot(initial_temp, -1 * initial_z, ls='--', color='k', label='obs start')
    plt.plot(final_temp_0, -1 * final_z_0, color='k', label='obs end')
    plt.ylabel('depth')
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.xlim(-2, 1)
    plt.ylim(-300, 0)
    plt.xlabel('temperature')
    plt.legend(loc=3)
    plt.savefig(str(save_path[j])+"/" + str(iteration).zfill(4) + '.png')
    plt.close()
    plt.clf()
    return 0

params = {
    #'font.size' : 32,
    #'axes.labelsize': 32,
    #'legend.fontsize': 32,
    #'xtick.labelsize': 32,
    #'ytick.labelsize': 32,
    #'text.usetex': False,
    'figure.figsize': [6,3 ],
    #'axes.linewidth' : 1.5
}
plt.rcParams.update(params)


##  load observations  ##

"""
profile_input_file = "seal55_ctd12.mat"
initial_profile = sio.loadmat(profile_input_file)
initial_z = initial_profile['profile']['z'][0, 0][:, 0]
initial_temp = initial_profile['profile']['t'][0, 0][:, 0]
initial_salt = initial_profile['profile']['s'][0, 0][:, 0]
initial_oxy = np.zeros(len(initial_z))
initial_density=pypwp.density_0(initial_temp,initial_salt)

lat=57

profile_input_file = "57_final.npz"
initial_profile = np.load(profile_input_file)
final_z_0 =initial_profile['z']
final_temp_0 = initial_profile['temp']
final_salt_0 = initial_profile['salt']
final_density_0=pypwp.density_0(final_temp_0,final_salt_0)




"""
profile_input_file = "pre_ice.npz"
initial_profile=np.load(profile_input_file)
initial_z=initial_profile['depth']
initial_salt=initial_profile['salt']
initial_temp=initial_profile['temp']
lat = 60
profile_input_file = "post_ice.npz"
initial_profile=np.load(profile_input_file)
final_z_0=initial_profile['depth']
final_salt_0=initial_profile['salt']
final_temp_0=initial_profile['temp']
initial_density=pypwp.density_0(initial_temp,initial_salt)
final_density_0=pypwp.density_0(final_temp_0,final_salt_0)


filenum=83  # 84 for 60 148 for 57
mli=251     #196 for 60 201 for 57
dz=3.0
z = np.arange(0,mli ) * dz

gridded_initial_temp=griddata(initial_z,initial_temp,z)
gridded_initial_salt=griddata(initial_z,initial_salt,z)



gridded_final_salt=griddata(final_z_0,final_salt_0,z)
gridded_final_temp=griddata(final_z_0,final_temp_0,z)


gridded_initial_temp[np.isnan(gridded_initial_temp)]=initial_temp[0]
gridded_initial_salt[np.isnan(gridded_initial_salt)]=initial_salt[0]


gridded_final_temp[np.isnan(gridded_final_temp)]=final_temp_0[0]
gridded_final_salt[np.isnan(gridded_final_salt)]=final_salt_0[0]

gridded_final_density=pypwp.density_0(gridded_final_temp,gridded_final_salt)
gridded_initial_density=pypwp.density_0(gridded_initial_temp,gridded_initial_salt)


fwc_initial=sum(dz*(34.8-gridded_initial_salt)/34.8)
fwc_final=sum(dz*(34.8-gridded_final_salt)/34.8)




print(fwc_initial,fwc_final)

#base_paths=['soccom_test_A_0.2_1/','soccom_test_A_0.5_1/','soccom_test_A_0.7_0/','soccom_test_A_0.89_0/']
#base_paths=['soccom_test_ag77_am6_0/','soccom_test_ag9_am7_0/','soccom_test_ag9_am2_0/']
#base_paths=['soccom_test_snow_A_0.2/','soccom_test_snow_A_0.5/','soccom_test_snow_A_0.7/','soccom_test_snow_A_0.89/']
#base_paths=['soccom_test_snow_ag77_am6/','soccom_test_snow_ag9_am7/','soccom_test_snow_ag9_am2/']
#base_paths=['soccom_v3_0.0/','soccom_test_v3_diff_0.1_2/','soccom_test_v3_diff_0.5_0/','soccom_test_v3_diff_5.0_0/','soccom_test_v3_diff_10.0/']
base_paths=['soccom_v3_0.0/']#,'soccom_v3_0.2/','soccom_v3_-0.2/']#,'soccom_v3_0.4/','soccom_v3_-0.4/','soccom_v3_0.8/','soccom_v3_-0.8/']
#base_paths=['soccom_v3_rb_0.5/','soccom_v3_rb_0.6/','soccom_v3_rb_0.7/','soccom_v3_0.0/','soccom_v3_rb_0.9/']
#base_paths=['soccom_v0_test_A_0.2_0/','soccom_v0_test_A_0.5_0/','soccom_v0_test_A_0.7_0/','soccom_v0_test_A_0.89_0/']
#base_paths=['soccom_v0_test_ag77_am6_0/','soccom_v0_test_ag9_am7_0/','soccom_v0_test_ag9_am2_0/']
base_paths=['soccom_v3_A0_09_hi_0.1/','soccom_v3_A0_09_hi_0.3/','soccom_v3_A0_09_hi_0.7/','soccom_v3_A0_09_hi_0.9/']
#base_paths=['soccom_v3_hi_05_A0_0.25_0/','soccom_v3_hi_05_A0_0.5_0/','soccom_v3_hi_05_A0_0.75_0/','soccom_v3_hi_05_A0_0.95_0/']
#base_paths=['soccom_test_A_0.2_1/','soccom_test_A_0.5_1/','soccom_test_A_0.7_0/','soccom_test_A_0.89_0/','soccom_test_ag77_am6_0/','soccom_test_ag9_am7_0/','soccom_test_ag9_am2_0/','soccom_test_snow_A_0.2/','soccom_test_snow_A_0.5/','soccom_test_snow_A_0.7/','soccom_test_snow_A_0.89/','soccom_test_snow_ag77_am6/','soccom_test_snow_ag9_am7/','soccom_test_snow_ag9_am2/'
            #,'soccom_v3_0.0/','soccom_test_v3_diff_0.1_2/','soccom_test_v3_diff_0.5_0/','soccom_test_v3_diff_5.0_0/','soccom_test_v3_diff_10.0/','soccom_v3_0.0/','soccom_v3_0.2/','soccom_v3_-0.2/','soccom_v3_0.4/','soccom_v3_-0.4/','soccom_v3_0.8/','soccom_v3_-0.8/'
            #,'soccom_v3_rb_0.5/','soccom_v3_rb_0.6/','soccom_v3_rb_0.7/','soccom_v3_0.0/','soccom_v3_rb_0.9/']#,'soccom_v0_test_A_0.2_0/','soccom_v0_test_A_0.5_0/','soccom_v0_test_A_0.7_0/','soccom_v0_test_A_0.89_0/','soccom_v0_test_ag77_am6_0/','soccom_v0_test_ag9_am7_0/','soccom_v0_test_ag9_am2_0/']

#base_paths=['soccom_v3__diff1e5_hi_05_A_0.25/','soccom_v3__diff1e5_hi_05_A_0.5/','soccom_v3__diff1e5_hi_05_A_0.75/','soccom_v3__diff1e5_hi_05_A_0.95/']
#base_paths=['soccom_v3_noor_diff1e5_hi_05_A_0.25/','soccom_v3_noor_diff1e5_hi_05_A_0.5/','soccom_v3_noor_diff1e5_hi_05_A_0.75/','soccom_v3_noor_diff1e5_hi_05_A_0.95/']
#base_paths=['soccom_no_ice_pwp/','soccom_no_ice_pwpkt/','soccom_no_ice_kt/','soccom_ice_ref/']
base_paths=['soccom_ice_dz_1.0/','soccom_ice_dz_2.0/','soccom_ice_ref/','soccom_ice_dz_5.0/']
base_paths=['soccom_ice_lat_0.8/','soccom_ice_lat_0.7/','soccom_ice_lat_0.6/','soccom_ice_lat_0.5/']
base_paths=['soccom_ice_snow_0.05/','soccom_ice_snow_0.1/','soccom_ice_snow_0.15/','soccom_ice_snow_0.2/']
base_paths=['soccom_ice_ref/','soccom_ice_advect_-0.1/','soccom_ice_advect_-0.2/','soccom_ice_advect_-0.4/','soccom_ice_advect_-0.6/','soccom_ice_advect_-0.8/']
base_paths=['soccom_ice_h_i0_0.1/','soccom_ice_h_i0_0.3/','soccom_ice_h_i0_0.7/','soccom_ice_h_i0_0.5/']

base_paths=['soccom_ice_ref/','soccom_ice_advect_0.1/','soccom_ice_advect_0.2/','soccom_ice_advect_0.4/','soccom_ice_advect_0.6/','soccom_ice_advect_0.8/']


savename='advect+'
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

final_temp=np.zeros((mli,len(base_paths)))
final_salt=np.zeros((mli,len(base_paths)))
final_density=np.zeros((mli,len(base_paths)))
temp=np.zeros((mli,filenum+1,len(base_paths)))
salt=np.zeros((mli,filenum+1,len(base_paths)))
density=np.zeros((mli,filenum+1,len(base_paths)))

z0=np.arange(mli)*3.
z1=np.arange(mli)*3.


for i in range(0,len(data_path)):
    scalars=np.load(os.path.join(data_path[i],'scalars.npz'))
    mld=scalars['mld']
    time=scalars['time']
    plt.plot(time,mld,lw=2,label=base_paths[i])

plt.title('soccom'+str(savename))
plt.xlabel("Days")
plt.ylabel("ML Depth")
plt.legend()
plt.tight_layout()
plt.savefig(str(comp_plot_output)+str(savename)+'/mld.png')
plt.clf()
ice_S_content=np.zeros(len(base_paths))
for i in range(0,len(data_path)):
    scalars=np.load(os.path.join(data_path[i],'scalars.npz'))
    h_i=scalars['hi']
    time=scalars['time']
    ice_S_content[i]=h_i[-1]*5
    plt.plot(time,h_i,lw=2,label=base_paths[i])

plt.title(str(savename))
plt.xlabel("Days")
plt.ylabel("ice thickness")
plt.legend()
plt.tight_layout()
plt.savefig(str(comp_plot_output)+str(savename)+'/hi.png')
plt.clf()

for i in range(0,len(data_path)):
    scalars=np.load(os.path.join(data_path[i],'scalars.npz'))
    A=scalars['A']
    time=scalars['time']
    plt.plot(time,A,lw=2,label=base_paths[i])

plt.title(str(savename))
plt.xlabel("Days")
plt.ylabel("ice concentration")
plt.legend()
plt.tight_layout()
plt.savefig(str(comp_plot_output)+str(savename)+'/A.png')
plt.clf()
"""
for i in range(0,len(data_path)):
    scalars=np.load(os.path.join(data_path[i],'scalars.npz'))
    emp=scalars['emp']
    isf=scalars['ice_salt_flux']
    time=scalars['time']
    plt.plot(time,emp,lw=2,label='emp')
    plt.plot(time, isf, lw=2, label='ice')

plt.title(str(savename))
plt.xlabel("Days")
plt.ylabel("ice concentration")
plt.legend()
plt.tight_layout()
plt.savefig(str(comp_plot_output)+str(savename)+'/salt_flux.png')
plt.clf()




"""
# load data into arrays and optionally plot individual profiles


for j in range(len(base_paths)):
    for i in range(0, filenum + 1):
        f_name = 'profiles_' + str(i) + '.0.npz'
        file_temp = np.load(os.path.join(data_path[j], f_name))
        temp_temp=file_temp['temp']
        salt_temp = file_temp['salt']
        density_temp = file_temp['density']
        temp[:,i,j]=temp_temp
        salt[:,i,j]=salt_temp
        density[:,i,j]=density_temp
        #plot_tsd(temp[:,i,j],salt[:,i,j],density[:,i,j],i) # plot profiles
        if i==filenum:
            final_temp[:,j]=temp_temp
            final_salt[:,j]=salt_temp
            final_density[:,j]=density_temp

file = open('metrics.txt',"w")
file.write('latitude = ' +str(lat) + ', FWC_initial = ' +str(fwc_initial)+ 'FWC final = ' + str(fwc_final)+'\n')
file.write(' Filename , fwc, RSS_temp , RSS salt , RSS Density' + '\n')
for j in range(len(base_paths)):
    fwc=sum(dz * ((34.8-final_salt[:,j]) / 34.8))+ice_S_content[j]
    temp_temp=final_temp[:,j]
    temp_salt=final_salt[:,j]
    temp_density=final_density[:,j]
    RSS_temp=(gridded_final_temp-temp_temp)**2
    #print(RSS_temp,gridded_final_temp)
    RSS_salt = (gridded_final_salt[0:100] - temp_salt[0:100])**2
    RSS_density=(gridded_final_density[0:100]- temp_density[0:100])**2
    #plt.plot(RSS_temp[0:100],z[0:100],label='temp RSS')
    #plt.plot(RSS_salt[0:100], z[0:100],label='salt RSS')
    #plt.plot(RSS_density[0:100], z[0:100],label='density RSS')
    #plt.title( "temp rss = %5.2f" % sum(RSS_temp) + ', salt rss = %5.2f ' % sum(RSS_salt)+ ' , density rss = %5.2f ' % sum(RSS_density))
    #plt.legend()
    #plt.savefig(str(comp_plot_output)+str(savename)+'/rss.png'))
    #plt.show()
    #plt.clf()
    file.write(str(base_paths[j])+ ' , ' + str(fwc)+ ' , ' + str(sum(RSS_temp)) + ' , ' + str(sum(RSS_salt))+' , ' + str(sum(RSS_density))+'\n')

file.close()
#plt.clf()

for j in range(len(base_paths)):
    temp_temp=final_temp[:,j]
    RSS_temp=(gridded_final_temp-temp_temp)**2
    plt.plot(RSS_temp[0:100],z[0:100],label=str(base_paths[j]))
plt.title("temp_rss")
plt.legend()
plt.savefig((str(comp_plot_output)+str(savename)+'/temp_rss.png'))
#plt.show()
plt.clf()

for j in range(len(base_paths)):
    temp_salt = final_salt[:, j]
    RSS_salt = (gridded_final_salt[0:100] - temp_salt[0:100]) ** 2
    plt.plot(RSS_salt[0:100], z[0:100], label=str(base_paths[j]))
plt.title("salt rss ")
plt.legend()
plt.savefig((str(comp_plot_output) +str(savename)+'/salt_rss.png'))
#plt.show()
plt.clf()

for j in range(len(base_paths)):
    temp_density=final_density[:,j]
    RSS_density = (gridded_final_density[0:100] - temp_density[0:100]) ** 2
    plt.plot(RSS_salt[0:100], z[0:100], label=str(base_paths[j]))
plt.title("density rss")
plt.legend()

plt.savefig((str(comp_plot_output)+str(savename) +'/density_rss.png'))
#plt.show()
plt.clf()
## Plot final profiles against the observational profile

plt.clf()
fig = plt.figure()

ax1 = fig.add_subplot(131)
plt.plot(initial_density, -1 * initial_z,ls='--',color='k', label='obs start ')
plt.plot(final_density_0, -1 * final_z_0,color='k', label='obs end')
plt.xlim(1027.4, 1027.8)
plt.ylim(-300,0)
plt.xlabel('density kg/m^3')
plt.ylabel('depth')
plt.xlim(1027.2,1027.9)
plt.xticks([1027.2,1027.4,1027.6,1027.8])
#ax1.set_xticklabels(['81027.2', '1027.6','1027.9'])

ax2 = fig.add_subplot(132)
plt.plot(initial_salt, -1*initial_z,color='k',ls='--', label='obs start')
plt.plot(final_salt_0, -1 * final_z_0,color='k', label='obs end')

plt.setp(ax2.get_yticklabels(), visible=False)
plt.xlim(33.8, 34.8)
plt.ylim(-300,0)
plt.xlabel('salinity')

ax3 = fig.add_subplot(133)
plt.plot(initial_temp, -1*initial_z,ls='--',color='k', label='obs start')
plt.plot(final_temp_0, -1 * final_z_0,color='k', label='obs end')

plt.setp(ax3.get_yticklabels(), visible=False)
plt.xlabel('temperature')
plt.xlim(-2, 1)
plt.ylim(-300,0)

for j in range(len(base_paths)):
    ax1 = fig.add_subplot(131)
    plt.plot(final_density[:,j], -z0, label=str(base_paths[j]))

    ax2 = fig.add_subplot(132)
    plt.plot(final_salt[:,j], -z0, label=str(base_paths[j]))

    ax3 = fig.add_subplot(133)
    plt.plot(final_temp[:,j], -z0, label=str(base_paths[j]))

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.figlegend(by_label.values(), by_label.keys(),loc='upper center',bbox_to_anchor=(0.5, 0.175),ncol=3)
plt.gcf().subplots_adjust(bottom=0.3)

#plt.tight_layout()
#plt.show()
plt.savefig(str(comp_plot_output)+str(savename)+'/end_profs.png')   #todo make this a specified folder
plt.close()
plt.clf()

## create pcolor plots of salt and heat with ml and hi*50 plotted on top
for j in range(len(base_paths)):
    scalars=np.load(os.path.join(data_path[j],'scalars.npz'))
    mld=scalars['mld']
    h_i=scalars['hi']
    #A=scalars['A']
    h_i_max=np.argmax(h_i)
    h_i_plot=-h_i*(50.)
    #A_plot=A*50
    depth=np.arange(mli)*3

    t=np.arange(len(mld))*1600.*100./8.64E4  #dt =100 here change 100 to whatever sim dt was.
    plt.figure(figsize = (20,10))
    plt.imshow(salt[0:68,:,j],extent=[0,t[-1],-depth[68],0],aspect=0.1) # imshow plots a heatmap
    plt.colorbar()
    plt.title(str(base_paths[j])+' salt')
    plt.tight_layout()
    plt.plot(t,-mld,label="ML")
    plt.plot(t,h_i_plot,label="hi*50")
    plt.xlabel("Days")
    plt.ylabel("Depth")
    plt.legend()
    #plt.plot(t,A_plot,label="A*50")
    #plt.show()
    plt.savefig(str(comp_plot_output)+str(savename)+'/'+str(base_paths[j][0:-1])+"_salt_heat.png")
    plt.clf()
    plt.figure(figsize = (20,10))
    plt.imshow(temp[0:68,:,j],extent=[0,t[-1],-depth[68],0],aspect=0.1) # imshow plots a heatmap
    plt.colorbar()
    plt.title(str(base_paths[j])+' temp')
    plt.tight_layout()
    plt.plot(t,-mld,label="ML")
    plt.plot(t,h_i_plot,label="hi*50")
    plt.xlabel("Days")
    plt.ylabel("Depth")
    plt.legend()
    #plt.plot(t,A_plot,label="A*50")

    #plt.show()
    plt.savefig(str(comp_plot_output)+str(savename)+'/'+str(base_paths[j][0:-1])+"_temp_heat.png")
