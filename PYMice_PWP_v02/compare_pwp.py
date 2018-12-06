import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import os
def plot_tsd(t0, s0, d0, iteration):
    plt.clf()
    # print(OLR_ice, ILR_ice, ISW_ice, i_sens, i_lat, T_si)
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    plt.title('d,s,t at %.2f days' % iteration)
    plt.plot(z0, d0, label='py')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel('density kg/m^3')
    ax2 = fig.add_subplot(312)
    plt.plot(z0, s0, label='py')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylabel('salinity')
    ax3 = fig.add_subplot(313)
    plt.plot(z0, t0, label='py')
    plt.xlabel('depth')
    plt.ylabel('temperature')
    plt.legend()
    plt.savefig(str(save_path)+"/" + str(i).zfill(4) + '.png')
    plt.close()
    print(i)
    return 0




filenum=540
##   load pypwp run
base_path='pypwp_20181205_0806/'

data_path =base_path+"data/"
save_path = str(base_path)+"/plots"    # tell it where to save the plots to be made
if not os.path.exists(save_path):
    os.makedirs(save_path)

## initialize arrays
temp=np.zeros((1501,filenum))
salt=np.zeros((1501,filenum))
density=np.zeros((1501,filenum))
z0=np.arange(1501)*3.
z1=np.arange(1501)*3.

for i in range(0,filenum):
    f_name = 'profiles_' + str(i) + '.npz'
    file_temp = np.load(os.path.join(data_path, f_name))
    temp_temp=file_temp['temp']
    salt_temp = file_temp['salt']
    density_temp = file_temp['density']
    temp[:,i]=temp_temp
    salt[:,i]=salt_temp
    density[:,i]=density_temp
    #plot_tsd(temp[:,i],salt[:,i],density[:,i],i)

scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld=scalars['mld']
h_i=scalars['hi']
#A=scalars['A']
we=scalars['we']
h_i_plot=h_i*50
depth=np.arange(251)*3
plt.figure(figsize = (20,10))
plt.imshow(temp,vmin=-2,vmax=1,extent=[0,filenum,750,0]) # imshow plots a heatmap
plt.colorbar()
t=np.arange(len(mld))
plt.plot(t,mld)
plt.plot(t,h_i_plot)
plt.savefig("test.png")

#plt.show()
plt.clf()

##   just note python does weird things with both savefig and show on so comment one out if using the other.
#plt.savefig("test.png")

plt.plot(t,mld)
plt.show()
plt.clf()
plt.plot(t,h_i)
plt.show()