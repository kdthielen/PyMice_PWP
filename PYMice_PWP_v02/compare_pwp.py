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
base_path0='bc-or-mli_1/'

data_path =base_path0+"data/"
save_path = str(base_path0)+"/plots"    # tell it where to save the plots to be made
if not os.path.exists(save_path):
    os.makedirs(save_path)

scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld=scalars['mld']
h_i=scalars['hi']

base_path1='bc-or-adi_1/'
data_path =base_path1+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld1=scalars['mld']
h_i1=scalars['hi']



base_path2='ktall_dz_2.0/'
data_path =base_path2+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld2=scalars['mld']
h_i2=scalars['hi']

base_path3='ktall_dz_3.0/'
data_path =base_path3+"data/"
scalars=np.load(os.path.join(data_path,'scalars.npz'))
mld3=scalars['mld']
h_i3=scalars['hi']

t=np.arange(len(mld))


##   just note python does weird things with both savefig and show on so comment one out if using the other.
#plt.savefig("test.png")


labels=['ad=mli','ad=adi','dz = 2.0 m','dz = 3.0 m']
plt.plot(t,mld,lw=2,label=labels[0])
plt.plot(t,mld1,lw=2,label=labels[1])
#plt.plot(t,mld2,lw=2,label=labels[2])
#plt.plot(t,mld3,lw=2,label=labels[3])
plt.xlabel("Days")
plt.ylabel("ML Depth")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig("ad3_q_ml.png")

plt.clf()

plt.plot(t,h_i,lw=2,label=labels[0])
plt.plot(t,h_i1,lw=2,label=labels[1])
#plt.plot(t,h_i2,lw=2,label=labels[2])
#plt.plot(t,h_i3,lw=2,label=labels[3])
plt.xlabel("Days")
plt.ylabel("Ice Thickness")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig("ad3_q_hi.png")

