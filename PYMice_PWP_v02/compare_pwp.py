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
base_paths=['noadvect_we_0/','advect_v3_-0.1_we/','advect_v3_-0.2_we/','advect_v3_-0.4_we/','advect_v3_-0.6_we/','advect_v3_-0.8_we/']
save_name="chloe_v3"
base_path0=base_paths[0]

data_path =base_path0+"data/"
save_path = str(base_path0)+"/plots"    # tell it where to save the plots to be made
if not os.path.exists(save_path):
    os.makedirs(save_path)

scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld0=scalars['mld']
h_i0=scalars['hi']
A0=scalars['A']
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


base_path1=base_paths[1]
data_path =base_path1+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld1=scalars['mld']
h_i1=scalars['hi']
A1=scalars['A']

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




base_path2=base_paths[2]
data_path =base_path2+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld2=scalars['mld']
h_i2=scalars['hi']
A2=scalars['A']

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


base_path3=base_paths[3]
data_path =base_path3+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld3=scalars['mld']
h_i3=scalars['hi']
A3=scalars['A']

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

base_path4=base_paths[4]
data_path =base_path4+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld4=scalars['mld']
h_i4=scalars['hi']
A4=scalars['A']

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


base_path5=base_paths[5]
data_path =base_path5+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld5=scalars['mld']
h_i5=scalars['hi']
A5=scalars['A']


temp_ml5=np.zeros(len(mld5))
salt_ml5=np.zeros(len(mld5))
density_ml5=np.zeros(len(mld5))

for i in range(0,len(mld5)):
    f_name = 'profiles_' + str(i) + '.0.npz'
    file_temp = np.load(os.path.join(data_path, f_name))
    temp_temp=file_temp['temp']
    salt_temp = file_temp['salt']
    density_temp = file_temp['density']
    temp_ml5[i]=temp_temp[0]
    salt_ml5[i]=salt_temp[0]
    density_ml5[i]=density_temp[0]


t0=np.arange(len(mld0))*1600.*100./8.64E4
t1=np.arange(len(mld1))*1600.*100./8.64E4
t2=np.arange(len(mld2))*1600.*100./8.64E4
t3=np.arange(len(mld3))*1600.*100./8.64E4







##   just note python does weird things with both savefig and show on so comment one out if using the other.
#plt.savefig("test.png")

plt.title('60 S 0 E chloe forcing')
labels=['-0.1','-0.2','-0.3','-0,4','-0.5','-0.6','']
plt.plot(t0,mld0,lw=2,label=base_paths[0])
plt.plot(t1,mld1,lw=2,label=base_paths[1])
plt.plot(t2,mld2,lw=2,label=base_paths[2])
plt.plot(t3,mld3,lw=2,label=base_paths[3])
plt.plot(t0,mld4,lw=2,label=base_paths[4])
#plt.plot(t0,mld5,lw=2,label=labels[5])


plt.xlabel("Days")
plt.ylabel("ML Depth")
plt.legend()
plt.tight_layout()
#plt.show()
#plt.savefig("60s_2015_compicekdtv3_rb_ml.png")
plt.savefig(str(save_name)+"_comp_ml_depth.png")


plt.clf()
plt.plot(t0,A0,lw=2,label=base_paths[0])
plt.plot(t0,A1,lw=2,label=base_paths[1])
plt.plot(t0,A2,lw=2,label=base_paths[2])
plt.plot(t0,A3,lw=2,label=base_paths[3])
plt.plot(t0,A4,lw=2,label=base_paths[4])
plt.plot(t0,A5,lw=2,label=base_paths[5])


plt.xlabel("Days")
plt.ylabel("Concentration")
plt.legend()
plt.tight_layout()
#plt.show()
#plt.savefig("60s_2015_comp_ml.png")
plt.savefig(str(save_name)+"_comp_A.png")

plt.clf()
plt.title('60 S 0 E chloe forcing')

plt.plot(t0,h_i0,lw=2,label=base_paths[0])
plt.plot(t1,h_i1,lw=2,label=base_paths[1])
plt.plot(t2,h_i2,lw=2,label=base_paths[2])
plt.plot(t3,h_i3,lw=2,label=base_paths[3])
plt.plot(t0,h_i4,lw=2,label=base_paths[4])
plt.plot(t0,h_i5,lw=2,label=base_paths[5])


plt.xlabel("Days")
plt.ylabel("Ice Thickness")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig(str(save_name)+"_comp_hi.png")
######------------------------------------------
plt.clf()
plt.title('60 S 0 E chloe forcing')

plt.plot(t0,temp_ml0,lw=2,label=base_paths[0])
plt.plot(t0,temp_ml1,lw=2,label=base_paths[1])
plt.plot(t0,temp_ml2,lw=2,label=base_paths[2])
plt.plot(t0,temp_ml3,lw=2,label=base_paths[3])
plt.plot(t0,temp_ml4,lw=2,label=base_paths[4])
plt.plot(t0,temp_ml5,lw=2,label=base_paths[5])


plt.xlabel("Days")
plt.ylabel("ML temp")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig(str(save_name)+"_comp_mlt.png")

plt.clf()

########-----------------------------

plt.title('60 S 0 E chloe forcing')

plt.plot(t0,salt_ml0,lw=2,label=base_paths[0])
plt.plot(t0,salt_ml1,lw=2,label=base_paths[1])
plt.plot(t0,salt_ml2,lw=2,label=base_paths[2])
plt.plot(t0,salt_ml3,lw=2,label=base_paths[3])
plt.plot(t0,salt_ml4,lw=2,label=base_paths[4])
plt.plot(t0,salt_ml5,lw=2,label=base_paths[5])


plt.xlabel("Days")
plt.ylabel("ML salinity")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig(str(save_name)+"_comp_mls.png")

plt.clf()

########-----------------------------

plt.title('60 S 0 E chloe forcing')

plt.plot(t0,density_ml0,lw=2,label=base_paths[0])
plt.plot(t0,density_ml1,lw=2,label=base_paths[1])
plt.plot(t0,density_ml2,lw=2,label=base_paths[2])
plt.plot(t0,density_ml3,lw=2,label=base_paths[3])
plt.plot(t0,density_ml4,lw=2,label=base_paths[4])
plt.plot(t0,density_ml5,lw=2,label=base_paths[5])

plt.xlabel("Days")
plt.ylabel("ML Density")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig(str(save_name)+"_comp_mlp.png")
plt.clf()
max_temp=np.zeros(5)
max_salt=np.zeros(5)
max_density=np.zeros(5)
