import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import os
import pypwp_functions as pypwp

def plot_tsd(t0, s0, d0, iteration):
    plt.clf()
    # print(OLR_ice, ILR_ice, ISW_ice, i_sens, i_lat, T_si)
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    plt.title('d,s,t at %.2f days' % iteration)
    plt.plot(d0, -z0, label='py')
    plt.plot(initial_density, -1 * initial_z, label='obs')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel('density kg/m^3')
    ax2 = fig.add_subplot(132)
    plt.plot(s0, -z0, label='py')
    plt.plot(initial_salt, -1*initial_z, label='obs')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylabel('salinity')
    ax3 = fig.add_subplot(133)
    plt.plot(t0, -z0, label='py')
    plt.plot(initial_temp, -1*initial_z, label='obs')
    plt.xlabel('depth')
    plt.ylabel('temperature')
    plt.legend()
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

filenum=84
##   load pypwp run
profile_input_file = "/home/thielen/Desktop/ttest/southern_seal_PWP.mat"
initial_profile = sio.loadmat(profile_input_file)
initial_z = initial_profile['pres'][:, 1]
initial_temp = initial_profile['temp'][:, 1]
initial_salt = initial_profile['sal'][:, 1]
initial_oxy = np.zeros(len(initial_z))
initial_density=pypwp.density_0(initial_temp,initial_salt)

base_paths=['chloe_ag77_am6/','chloe_ag9_am7/','chloe_ag9_am2/']
data_path=[]
save_path=[]
i=0
for fname in base_paths:
    data_path.append(fname+"data/")
    save_path.append(fname+"plots")  # tell it where to save the plots to be made
    if not os.path.exists(save_path[i]):
        os.makedirs(save_path[i])
    i+=1

final_temp=np.zeros((196,len(base_paths)))
final_salt=np.zeros((196,len(base_paths)))
final_density=np.zeros((196,len(base_paths)))
## initialize arrays
temp=np.zeros((196,filenum+1,len(base_paths)))
salt=np.zeros((196,filenum+1,len(base_paths)))
density=np.zeros((196,filenum+1,len(base_paths)))
z0=np.arange(196)*3.
z1=np.arange(196)*3.

for i in range(0,filenum+1):
    for j in range(len(base_paths)):
        f_name = 'profiles_' + str(i) + '.0.npz'
        file_temp = np.load(os.path.join(data_path[j], f_name))
        temp_temp=file_temp['temp']
        salt_temp = file_temp['salt']
        density_temp = file_temp['density']
        temp[:,i,j]=temp_temp
        salt[:,i,j]=salt_temp
        density[:,i,j]=density_temp
        #plot_tsd(temp[:,i],salt[:,i],density[:,i],i)
        if i==filenum:
            final_temp[:,j]=temp_temp
            final_salt[:,j]=salt_temp
            final_density[:,j]=density_temp


plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(131)

plt.plot(initial_density, -1 * initial_z, label='obs')
plt.xlim(1027.4, 1027.8)
plt.ylim(-300,0)
plt.xlabel('density kg/m^3')
plt.ylabel('depth')
ax1.set_xticklabels(['1027.4', '1027.6','1027.8'])

ax2 = fig.add_subplot(132)
plt.plot(initial_salt, -1*initial_z, label='obs')
plt.setp(ax2.get_yticklabels(), visible=False)
plt.xlim(33.8, 34.8)
plt.ylim(-300,0)
plt.xlabel('salinity')
ax3 = fig.add_subplot(133)
plt.plot(initial_temp, -1*initial_z, label='obs')
plt.setp(ax3.get_yticklabels(), visible=False)
plt.xlabel('temperature')
plt.xlim(-2, 1)
plt.ylim(-300,0)
from collections import OrderedDict
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
plt.savefig('/home/thielen/Desktop/ttest/v2_chloe_const.png')
plt.close()
plt.clf()

for j in range(len(base_paths)):
    scalars=np.load(os.path.join(data_path[j],'scalars.npz'))
    mld=scalars['mld']
    h_i=scalars['hi']
    #A=scalars['A']
    h_i_max=np.argmax(h_i)
    h_i_plot=-h_i*(50.)
    #A_plot=A*50
    depth=np.arange(196)*3
    t=np.arange(len(mld))*1600.*100./8.64E4
    plt.figure(figsize = (20,10))
    plt.imshow(salt[:,:,j],extent=[0,t[-1],-196*3,0],aspect=0.1) # imshow plots a heatmap
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
    plt.savefig(os.path.join(save_path[j],"salt_heat.png"))
    plt.clf()
    plt.figure(figsize = (20,10))
    plt.imshow(temp[:,:,j],extent=[0,t[-1],-196*3,0],aspect=0.1) # imshow plots a heatmap
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
    plt.savefig(os.path.join(save_path[j],"temp_heat.png"))