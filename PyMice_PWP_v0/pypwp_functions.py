import numpy as np
import os as os
from params import *

def pure_water_eos(T):
    a0 = 999.842594
    a1 =   6.793952e-2
    a2 =  -9.095290e-3
    a3 =   1.001685e-4
    a4 =  -1.120083e-6
    a5 =   6.536332e-9
    dens = a0 + (a1 + (a2 + (a3 + (a4 + a5*T)*T)*T)*T)*T
    return dens

def density_0(T,S):
    b0 =  8.24493e-1
    b1 = -4.0899e-3
    b2 =  7.6438e-5
    b3 = -8.2467e-7
    b4 =  5.3875e-9
    c0 = -5.72466e-3
    c1 =  1.0227e-4
    c2 = -1.6546e-6
    d0 = 4.8314e-4
    dens = pure_water_eos(T) + (b0 + (b1 + (b2 + (b3 + b4*T)*T)*T)*T)*S + (c0 + (c1 + c2*T)*T)*S*(np.sqrt(S)) + d0*S**2.
    return dens

def liquidous(S):
    return -0.054*S

def saturation_spec_hum(ts):
    ew = 6.112*np.e**((17.62*(ts))/(243.12+ts))
    ew = ew*100.
    #Using AOMIP definition of specific humidity. Can't find description of where 0.378 comes from
    sat_sp_hum = (ep*ew)/(P_atm - (0.378*ew))
    return sat_sp_hum

#################
##   Diffusion ##  todo this does not permit double diffusion? need to understand where this comes from
#################

def diffusion(temp,salt,oxy,u,v,tracer_diff,oxy_diff,vel_diff,nz):
    dstab1 = tracer_diff #0.001;
    dstab2 = vel_diff #0.005;
    dstab3 = oxy_diff  #0.001;
    temp[1:nz-1] = temp[1:nz-1]+dstab1*(temp[0:nz-2]-2.*temp[1:nz-1]+temp[2:])
    salt[1:nz-1] = salt[1:nz-1]+dstab1*(salt[0:nz-2]-2.*salt[1:nz-1]+salt[2:])
    oxy[1:nz-1] = oxy[1:nz-1]+dstab3*(oxy[0:nz-2]-2.*oxy[1:nz-1]+oxy[2:])
    u[1:nz-1] = u[1:nz-1]+dstab2*(u[0:nz-2]-2.*u[1:nz-1]+u[2:])
    v[1:nz-1] = v[1:nz-1]+dstab2*(v[0:nz-2]-2.*v[1:nz-1]+v[2:])
    return temp,salt,oxy,u,v

#############################
##    Radiative Transfer   ##
#############################

def absorb(beta1,beta2,nz,dz):

#  Compute solar radiation absorption profile. This
#  subroutine assumes two wavelengths, and a double
#  exponential depth dependence for absorption.

#  Subscript 1 is for red, non-penetrating light, and
#  2 is for blue, penetrating light. rs1 is the fraction
#  assumed to be red.
    rs1 = 0.6
    rs2 = 1.0-rs1
    absrb = np.zeros(nz)
    z1 = np.arange(0,nz)*dz
    z2 = z1 + dz
    z1b1 = z1/beta1
    z2b1 = z2/beta1
    z1b2 = z1/beta2
    z2b2 = z2/beta2
    absrb = (rs1*(np.e**(-z1b1)-np.e**(-z2b1))+rs2*(np.e**(-z1b2)-np.e**(-z2b2)))
    return absrb


#############################
##  heat budget functions  ##
#############################


def lw_emission(T_in,emiss):
    return emiss*stef_boltz*((T_in+273.15)**4)

def lw_downwelling(lw,emiss):
    return emiss*lw

def sw_downwelling(sw,albedo): #think 1-IO here from petty saying this is all in their So so maybe remove
    return (1.-albedo)*sw

def ao_sens(T_so,T_a,U_a):
    return rho_air_ref*cp_air*C_turb_o*U_a*(T_so-T_a)

def ao_latent(T_so,U_a,sat_sp_hum,sp_hum):
    return rho_air_ref*Latent_vapor*C_turb_o*U_a*(sat_sp_hum-sp_hum)

#############################################
##   PWP Mixing Functions and Dependents   ##
#############################################

def mix(t,s,u,v,d,ml_index):
    t[0:ml_index+1] = np.mean(t[0:ml_index+1])
    s[0:ml_index+1] = np.mean(s[0:ml_index+1])
    d[0:ml_index+1] = density_0(t[0:ml_index+1],s[0:ml_index+1])
    u[0:ml_index+1] = np.mean(u[0:ml_index+1])
    v[0:ml_index+1] = np.mean(v[0:ml_index+1])
    return t,s,u,v,d

def remove_static_instabilities(ml_index,t,s,u,v,d): # if grid cell just below ML is lighter, mix into ML
    while d[ml_index] > d[ml_index+1]:
        t,s,u,v,d = mix(t,s,u,v,d,ml_index+1)
        ml_index = ml_index+1
    return t,s,u,v,d, ml_index

def rot(ang,u,v):
    r_temp = (u+v*1j)*np.e**(ang*1j)
    u_temp = np.real(r_temp)
    v_temp = np.imag(r_temp)
    return u_temp,v_temp


def stir(rc, r_temp, j, t, s, d, u, v, oxy):
    #  This subroutine mixes cells j and j+1 just enough so that
    #  the Richardson number after the mixing is brought up to
    #  the value rnew. In order to have this mixing process
    #  converge, rnew must exceed the critical value of the
    #  richardson number where mixing is presumed to start. If
    #  r critical = rc = 0.25 (the nominal value), and r = 0.20, then
    #  rnew = 0.3 would be reasonable. If r were smaller, then a
    #  larger value of rnew - rc is used to hasten convergence.

    #  This subroutine was modified by JFP in Sep 93 to allow for an
    #  aribtrary rc and to achieve faster convergence.
    rcon = 0.02 + (rc - r_temp) / 2.
    rnew = rc + rcon / 5.
    temporary_f = 1. - r_temp / rnew
    dtemp = (t[j + 1] - t[j]) * temporary_f / 2.
    t[j + 1] = t[j + 1] - dtemp
    t[j] = t[j] + dtemp
    ds = (s[j + 1] - s[j]) * temporary_f / 2.
    s[j + 1] = s[j + 1] - ds
    s[j] = s[j] + ds
    d[j] = density_0(t[j], s[j])
    d[j + 1] = density_0(t[j + 1], s[j + 1])
    du_temp = (u[j + 1] - u[j]) * temporary_f / 2.
    u[j + 1] = u[j + 1] - du_temp
    u[j] = u[j] + du_temp
    dv = (v[j + 1] - v[j]) * temporary_f / 2.
    v[j + 1] = v[j + 1] - dv
    v[j] = v[j] + dv
    do2 = (oxy[j + 1] - oxy[j]) * temporary_f/2.
    oxy[j + 1] = oxy[j + 1] - do2
    oxy[j] = oxy[j] + do2
    return t, s, d, u, v, oxy

def grad_mix(dz,g,rg,nz,z,t,s,d,u,v,oxy,ml_index): #PWP mixing based on gradient richardson number

#  This function performs the gradeint Richardson Number relaxation
#  by mixing adjacent cells just enough to bring them to a new
#  Richardson Number.
    rc 	= rg
#  Compute the gradeint Richardson Number, taking care to avoid dividing by
#  zero in the mixed layer.  The numerical values of the minimum allowable
#  density and velocity differences are entirely arbitrary, and should not
#  effect the calculations (except that on some occasions they evidnetly have!)
    check=1
    j1 = 0
    j2 = nz-2
    r_check=np.zeros(nz)+9999999
    while check==1:
        for j in range(j1,j2):
            dd = (d[j+1]-d[j])/d[j]
            dv = (u[j+1]-u[j])**2+(v[j+1]-v[j])**2
            if dv < 1E-15: #changed this back to 0 from 1e-10
                r_check[j] = 99999.
            else:
                r_check[j] = g*dd*dz/dv
        min_rg = np.min(r_check)  #todo need index as well
        if (min_rg>rc):		# Check to see whether the smallest r is critical or not.
            check=0
        else:
    #  Mix the cells js and js+1 that had the smallest Richardson Number
            min_ind = int(np.argmin(r_check))
            if type(min_ind)!=int:
                print(min_ind, min_rg)
                break
            js = min_ind
            t, s, d, u, v, oxy = stir(rc, min_rg, js, t, s, d, u, v, oxy)
            t, s, u, v, d, ml_index = remove_static_instabilities(ml_index, t, s, u, v, d)#  Recompute the Richardson Number over the part of the profile that has changed

            j1 = js-2
            if j1 < 0:
                j1 = 0
            j2 = js+2
            if (j2 > nz-2):
                j2 = nz-2
    return t, s, d, u, v, oxy, ml_index




def bulk_mix(ml_index,rb,d,u,v,t,s,z,nz): #mixing for PWP based on Bulk Richardson number arguements
    rvc = 0.65
    for j in range(ml_index+1,int(nz)):
        h 	= z[j]
        dd 	= (d[j]-d[0])/d[0]
        dv 	= (u[j]-u[0])**2+(v[j]-v[0])**2
        if dv < 1E-15:
            rv = 9999.
        else:
            rv = g*h*dd/dv #save rv below this in here and matlab and compare?
        if rv>rvc:
            break
        else:
            t, s, u, v, d= mix(t,s,u,v,d,j)
            #ml_index+=1
    return ml_index,d,u,v,t,s



def Ocean_relax(t,s,o,t_0,s_0,o_0,ad_i):
    dtemp = ((-0.01*(t[ad_i:]-t_0[ad_i:]))*0.01)	#fake advection below ad_i/relax to ocean state. not sure why times -0.001
    t[ad_i:] = t[ad_i:] + dtemp
    dsalt = ((-0.01*(s[ad_i:]-s_0[ad_i:]))*0.01)
    s[ad_i:] = s[ad_i:] + dsalt
    doxy = ((-0.01*(o[ad_i:]-o_0[ad_i:]))*0.01)
    o[ad_i:] = o[ad_i:] + doxy
    return t, s, o



def Oxygen_change(t,s,oxy,d,UU,ml_depth,ml_index,dt,A,): #todo turn on ice

    oxy_sat = sw_sat_oxy(s[0],t[0])
    oxy_sat = oxy_sat*44.658
    oxy_sat = oxy_sat/d[0]
    oxy_sat = oxy_sat*1000.
# #Calculate the schmidt number -- using values and equation from Wanninkhof 2014
    Sc = 1920.4+(-135.6*t[0])+(5.2122*(t[0]**2))+(-0.10939*(t[0]**3))+(0.00093777*(t[0]**4))
# #Calculated gas exchange rate
    kk = (1.791e-5)*((UU*UU)*(Sc**(-0.5)))
    #ddoxy = kk*(oxy[0] - oxy_sat)*(dt/ml_depth)

# #change in oxy
    if A ==0: #if no sea ice is present
        ddoxy = kk*(oxy[0] - oxy_sat)*(dt/ml_depth)
    else:
        ddoxy = (kk*0.25)*(oxy[0]-oxy_sat)*(dt/ml_depth) # if sea ice is present - see Loose et al. 2014

    oxy[0:ml_index+1] = oxy[0:ml_index+1] - ddoxy
    return oxy

def sw_sat_oxy(S,T):
    T = 273.15 + T * 1.00024
    a1 = -173.4292
    a2 = 249.6339
    a3 = 143.3483
    a4 = -21.8492
    b1 = -0.033096
    b2 = 0.014259
    b3 = -0.0017000

    # Eqn(4) of Weiss 1970
    lnC = a1+a2*(100./T)+a3*np.log(T/100.)+a4*(T/100.)+S*(b1+b2*(T/100.)+b3*((T/100.)**2))
    c = np.e**(lnC)
    return c



def write_params(kt_switch,bc_ice,dt,dz,days,depth,ad,dt_save,tracer_diff,oxy_diff,vel_diff,lat,rb,rg,rkz,m_kt,n_kt,cd_ice,cd_ocean,S_ice,h_ice_min,A_max,R_b,phi_r,T_si_0,base_path):
    var = os.path.join(base_path, 'variables.txt')
    f=open(var,'w')
    f.write('Kt switch' +str(kt_switch) +'\n')
    f.write('bc switch' + str(bc_ice) + '\n')
    f.write('dt = ' +str(dt) +'\n')
    f.write('dz = ' +str(dz) +'\n')
    f.write('days = ' +str(days) +'\n')
    f.write('depth = ' +str(depth) +'\n')
    f.write('ad = ' +str(ad) +'\n')
    f.write('ad_i = ' +str(ad_i) +'\n')
    f.write('dt_save = ' +str(dt_save) +'\n')
    f.write('tracer_diff = ' +str(tracer_diff) +'\n')
    f.write('oxy_diff = ' +str(oxy_diff) +'\n')
    f.write('vel_diff = ' +str(vel_diff) +'\n')
    f.write('lat = ' +str(lat) +'\n')
    f.write('rb = ' +str(rb) +'\n')
    f.write('rg = ' +str(rg) +'\n')
    f.write('rkz = ' +str(rkz) +'\n')
    f.write('m_kt = ' +str(m_kt) +'\n')
    f.write('n_kt = ' +str(n_kt) +'\n')
    f.write('cd_ice = ' +str(cd_ice) +'\n')
    f.write('cd_ocean = ' +str(cd_ocean) +'\n')
    f.write('S_ice = ' +str(S_ice) +'\n')
    f.write('h_ice_min = ' +str(h_ice_min) +'\n')
    f.write('A_max = ' +str(A_max) +'\n')
    f.write('R_b = ' +str(R_b) +'\n')
    f.write('phi_r = ' +str(phi_r) +'\n')
    f.write('T_si_0' + str(T_si_0) + '\n')
    f.close()
    return 0