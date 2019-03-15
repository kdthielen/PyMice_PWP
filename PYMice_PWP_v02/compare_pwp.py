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

params = {
    'font.size' : 12,
    'axes.labelsize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': [6,6 ],
    'axes.linewidth' : 1.5
}
plt.rcParams.update(params)


filenum=540
##   load pypwp run


base_path0='ham_22/'

data_path =base_path0+"data/"
save_path = str(base_path0)+"/plots"    # tell it where to save the plots to be made
if not os.path.exists(save_path):
    os.makedirs(save_path)

scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld0=scalars['mld']
h_i0=scalars['hi']

temp_ml0=np.zeros(len(mld0))
salt_ml0=np.zeros(len(mld0))
density_ml0=np.zeros(len(mld0))

for i in range(0,len(mld0)):
    f_name = 'profiles_' + str(i) + '.0.npz'
    file_temp = np.load(os.path.join(data_path, f_name))
    temp_temp=file_temp['temp']
    salt_temp = file_temp['salt']
    density_temp = file_temp['density']
    temp_ml0[i]=temp_temp[0]
    salt_ml0[i]=salt_temp[0]
    density_ml0[i]=density_temp[0]


base_path1='ham_23/'
data_path =base_path1+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld1=scalars['mld']
h_i1=scalars['hi']

temp_ml1=np.zeros(len(mld1))
salt_ml1=np.zeros(len(mld1))
density_ml1=np.zeros(len(mld1))

for i in range(0,len(mld1)):
    f_name = 'profiles_' + str(i) + '.0.npz'
    file_temp = np.load(os.path.join(data_path, f_name))
    temp_temp=file_temp['temp']
    salt_temp = file_temp['salt']
    density_temp = file_temp['density']
    temp_ml1[i]=temp_temp[0]
    salt_ml1[i]=salt_temp[0]
    density_ml1[i]=density_temp[0]




base_path2='icesw/'
data_path =base_path2+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld2=scalars['mld']
h_i2=scalars['hi']
#A2=scalars['A']

temp_ml2=np.zeros(len(mld2))
salt_ml2=np.zeros(len(mld2))
density_ml2=np.zeros(len(mld2))

for i in range(0,len(mld2)):
    f_name = 'profiles_' + str(i) + '.0.npz'
    file_temp = np.load(os.path.join(data_path, f_name))
    temp_temp=file_temp['temp']
    salt_temp = file_temp['salt']
    density_temp = file_temp['density']
    temp_ml2[i]=temp_temp[0]
    salt_ml2[i]=salt_temp[0]
    density_ml2[i]=density_temp[0]


base_path3='ham_5/'
data_path =base_path3+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld3=scalars['mld']
h_i3=scalars['hi']
#A3=scalars['A']

temp_ml3=np.zeros(len(mld3))
salt_ml3=np.zeros(len(mld3))
density_ml3=np.zeros(len(mld3))

for i in range(0,len(mld3)):
    f_name = 'profiles_' + str(i) + '.0.npz'
    file_temp = np.load(os.path.join(data_path, f_name))
    temp_temp=file_temp['temp']
    salt_temp = file_temp['salt']
    density_temp = file_temp['density']
    temp_ml3[i]=temp_temp[0]
    salt_ml3[i]=salt_temp[0]
    density_ml3[i]=density_temp[0]

base_path4='diff_adi_long_0.01/'
data_path =base_path4+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld4=scalars['mld']
h_i4=scalars['hi']
#A4=scalars['A']

temp_ml4=np.zeros(len(mld4))
salt_ml4=np.zeros(len(mld4))
density_ml4=np.zeros(len(mld4))

for i in range(0,len(mld4)):
    f_name = 'profiles_' + str(i) + '.0.npz'
    file_temp = np.load(os.path.join(data_path, f_name))
    temp_temp=file_temp['temp']
    salt_temp = file_temp['salt']
    density_temp = file_temp['density']
    temp_ml4[i]=temp_temp[0]
    salt_ml4[i]=salt_temp[0]
    density_ml4[i]=density_temp[0]


base_path5='dt_adi_long_200/'
data_path =base_path5+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld5=scalars['mld']
h_i5=scalars['hi']
#A5=scalars['A']


t=np.arange(len(mld0))*1600.*100./8.64E4


##   just note python does weird things with both savefig and show on so comment one out if using the other.
#plt.savefig("test.png")

plt.title('Variable dt, Relaxation Below 597')
labels=['10','25','50','200']
plt.plot(t,mld0,lw=2,label=labels[0])
plt.plot(t,mld1,lw=2,label=labels[1])
plt.plot(t,mld2,lw=2,label=labels[2])
plt.plot(t,mld3,lw=2,label=labels[3])
#plt.plot(t,mld4,lw=2,label=labels[4])
#plt.plot(t,mld5,lw=2,label=labels[5])


plt.xlabel("Days")
plt.ylabel("ML Depth")
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig("dt_adi_ml.png")
plt.clf()
#plt.plot(t,A0,lw=2,label=labels[0])
#plt.plot(t,A1,lw=2,label=labels[1])
#plt.plot(t,A2,lw=2,label=labels[2])
#plt.plot(t,A3,lw=2,label=labels[3])
#plt.plot(t,A4,lw=2,label=labels[4])
#plt.plot(t,A5,lw=2,label=labels[5])


#plt.xlabel("Days")
#plt.ylabel("ML Depth")
#plt.legend()
#plt.tight_layout()
#plt.show()
#plt.savefig("prog_compare_ml.png")

plt.clf()
plt.title('Variable dt, Relaxation Below 597')

plt.plot(t,h_i0,lw=2,label=labels[0])
plt.plot(t,h_i1,lw=2,label=labels[1])
plt.plot(t,h_i2,lw=2,label=labels[2])
plt.plot(t,h_i3,lw=2,label=labels[3])
#plt.plot(t,h_i4,lw=2,label=labels[4])
#plt.plot(t,h_i5,lw=2,label=labels[5])


plt.xlabel("Days")
plt.ylabel("Ice Thickness")
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig("dt_adi_hi.png")
######------------------------------------------
plt.clf()
plt.title('Variable Diffusion, Relaxation Below 597')

plt.plot(t,temp_ml0,lw=2,label=labels[0])
plt.plot(t,temp_ml1,lw=2,label=labels[1])
plt.plot(t,temp_ml2,lw=2,label=labels[2])
plt.plot(t,temp_ml3,lw=2,label=labels[3])
#plt.plot(t,temp_ml4,lw=2,label=labels[4])
#plt.plot(t,temp_ml5,lw=2,label=labels[5])


plt.xlabel("Days")
plt.ylabel("ML temp")
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig("dt_adi_ml.png")

plt.clf()

########-----------------------------

plt.title('Variable Diffusion Without Relaxation')

plt.plot(t,salt_ml0,lw=2,label=labels[0])
plt.plot(t,salt_ml1,lw=2,label=labels[1])
plt.plot(t,salt_ml2,lw=2,label=labels[2])
plt.plot(t,salt_ml3,lw=2,label=labels[3])
#plt.plot(t,salt_ml4,lw=2,label=labels[4])
#plt.plot(t,salt_ml5,lw=2,label=labels[5])


plt.xlabel("Days")
plt.ylabel("ML salinity")
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig("oroff_newdiff_MLS.png")

plt.clf()

########-----------------------------

plt.title('Variable Diffusion Without Relaxation')

plt.plot(t,density_ml0,lw=2,label=labels[0])
plt.plot(t,density_ml1,lw=2,label=labels[1])
plt.plot(t,density_ml2,lw=2,label=labels[2])
plt.plot(t,density_ml3,lw=2,label=labels[3])
#plt.plot(t,density_ml4,lw=2,label=labels[4])
#plt.plot(t,density_ml5,lw=2,label=labels[5])


plt.xlabel("Days")
plt.ylabel("ML Density")
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig("oroff_newdiff_mlp.png")
plt.clf()
max_temp=np.zeros(5)
max_salt=np.zeros(5)
max_density=np.zeros(5)
