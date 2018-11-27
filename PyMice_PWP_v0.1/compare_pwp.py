import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import os
def plot_tsd(t0,t1, s0,s1, d0,d1, iteration):
    plt.clf()
    # print(OLR_ice, ILR_ice, ISW_ice, i_sens, i_lat, T_si)
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    plt.title('d,s,t at %.2f days' % iteration)
    plt.plot(z0, d0, label='py')
    plt.plot(z1,d1,label='mat')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel('density kg/m^3')
    ax2 = fig.add_subplot(312)
    plt.plot(z0, s0, label='py')
    plt.plot(z1, s1, label='mat')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylabel('salinity')
    ax3 = fig.add_subplot(313)
    plt.plot(z0, t0, label='py')
    plt.plot(z1, t1, label='mat')
    plt.xlabel('depth')
    plt.ylabel('temperature')
    plt.legend()
    plt.savefig(str(save_path)+"/" + str(i).zfill(4) + '.png')
    plt.close()
    print(i)
    return 0




##   load matlab file to compare with
matlab_out_file='test.mat'
print('loading ', str(matlab_out_file))

mat_data = sio.loadmat(matlab_out_file)
ml_series = mat_data['pwp_output']['ml'][0,0][0,:] # n
hi_series = mat_data['pwp_output']['hi'][0,0][0,:] #
time_series = mat_data['pwp_output']['time'][0,0][0,:] #
temp_series=mat_data['pwp_output']['t'][0,0] # this is a z x t array
salt_series=mat_data['pwp_output']['s'][0,0] # this is a z x t array
density_series=mat_data['pwp_output']['d'][0,0] # this is a z x t array
we_series = mat_data['pwp_output']['we'][0,0][0,:]  #mpwp has these saved unless altered- if non existentin mat run, comment/delete
Pw_series = mat_data['pwp_output']['Pw'][0,0][0,:]
Pb_series = mat_data['pwp_output']['Pb'][0,0][0,:]
#mr_series = mat_data['pwp_output']['mr'][0,0][0,:]
filenum= len(ml_series) #can just set to num of save points in py sim. (can be automated using some os packages to check how many savepoints there are.

##   load pypwp run
base_path='pypwp_20181127_1308/'

data_path =base_path+"data/"
save_path = str(base_path)+"/plots"    # tell it where to save the plots to be made
if not os.path.exists(save_path):
    os.makedirs(save_path)

## initialize arrays
temp=np.zeros((251,filenum))
salt=np.zeros((251,filenum))
density=np.zeros((251,filenum))
z0=np.arange(251)*3.
z1=np.arange(251)*3.

for i in range(0,filenum):
    f_name = 'profiles_' + str(i) + '.npz'
    file_temp = np.load(os.path.join(data_path, f_name))
    temp_temp=file_temp['temp']
    salt_temp = file_temp['salt']
    density_temp = file_temp['density']
    temp[:,i]=temp_temp
    salt[:,i]=salt_temp
    density[:,i]=density_temp
    #plot_tsd(temp[:,i],temp_series[:,i],salt[:,i],salt_series[:,i],density[:,i],density_series[:,i],i)

scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld=scalars['mld']
h_i=scalars['hi']
h_i_plot=h_i*50.
depth=np.arange(251)*3
plt.figure(figsize = (20,10))
plt.imshow(temp,vmin=-2,vmax=1,extent=[0,5000,750,0]) # imshow plots a heatmap
plt.colorbar()
t=np.arange(len(mld))

plt.plot(t,mld)
plt.plot(t,h_i_plot)

##   just note python does weird things with both savefig and show on so comment one out if using the other.
#plt.savefig("test.png")
#plt.show()
plt.clf()
t=np.arange(len(ml_series))

plt.plot(t,ml_series)
plt.plot(t,mld[0:len(ml_series)])
plt.show()

