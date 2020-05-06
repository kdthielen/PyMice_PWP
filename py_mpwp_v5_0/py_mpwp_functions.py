import numpy as np
from numba import jit

jit_no_python = False

# thermodynamics
sw_a0 = 999.842594
sw_a1 =   6.793952e-2
sw_a2 =  -9.095290e-3
sw_a3 =   1.001685e-4
sw_a4 =  -1.120083e-6
sw_a5 =   6.536332e-9
sw_b0 =  8.24493e-1
sw_b1 = -4.0899e-3
sw_b2 =  7.6438e-5
sw_b3 = -8.2467e-7
sw_b4 =  5.3875e-9
sw_c0 = -5.72466e-3
sw_c1 =  1.0227e-4
sw_c2 = -1.6546e-6
sw_d0 = 4.8314e-4

sw_sat_oxy_a1 = -173.4292
sw_sat_oxy_a2 = 249.6339
sw_sat_oxy_a3 = 143.3483
sw_sat_oxy_a4 = -21.8492
sw_sat_oxy_b1 = -0.033096
sw_sat_oxy_b2 = 0.014259
sw_sat_oxy_b3 = -0.0017000

# physical constants
P_atm           = 101325.
rho_air_ref     =1.275
rho_ice_ref	    = 930.		# ice density
rho_ocean_ref	= 1026.		# ocean reference

cp_air          = 1005.
cp_ocean	    = 4190.		#

Latent_fusion	= 3.340e5
Latent_vapor    = 2.501e6
Latent_sub      = 2.834e6

##############################################
##             Sea Ice Settings             ##
##############################################
A_min       = 0     # minimum sea ic concentration
A_melt      = 0.    # lower limit of sea ice concentration.
A_grow      = 0.95  # upper limit of sea ice concentration.
h_ice_min	= 0.    # minimum ice thickness
R_b         = 0.8   # lateral vs basal melting paramater (1 for all basal )
S_ice		= 5     # average bulk salinity of sea ice

C_turb_i        = 0.0013	# Ebert and Curry 1993 turbulent exchange over ice
C_turb_o	    = 0.001		# Ebert and Curry 1993 turbulent exchange over leads
Stanton		    = 0.006		# McPhee 1992 mixed layer to sea ice heat transfer

cond_on         = 1     #f or conductivity term in heat balance of full_ice- implemented mainly for testing
cond_ice	    = 2.04		# cond of ice
cond_snow	    = 0.31		# conductivity value for snow

snow_emiss	    = 0.99		#
snow_albedo	    = 0.8		#
ice_emiss       = 0.99      # EBERT
ice_albedo      = 0.63 


def init_eos(equation_of_state='eos-80',alpha0=0.5e-4,beta0=8.e-4):
    globals().update({'alpha0':alpha0,'beta0':beta0,'equation_of_state':equation_of_state})

@jit(nopython=jit_no_python)
def density_0(T, S):
    if equation_of_state=='eos-80':
        density = sw_a0 + (sw_a1 + (sw_a2 + (sw_a3 + (sw_a4 + sw_a5*T)*T)*T)*T)*T + \
            (sw_b0 + (sw_b1 + (sw_b2 + (sw_b3 + sw_b4*T)*T)*T)*T)*S + \
            (sw_c0 + (sw_c1 + sw_c2*T)*T)*S*(np.sqrt(S)) + sw_d0*S**2.
    elif equation_of_state=='linear':
        density = rho_ocean_ref*(1-alpha0*(T-10.)+beta0*(S-35.))
    return density

# Thermal expansion coefficient
@jit(nopython=jit_no_python)
def sw_alpha(T,S):
    alpha = sw_a1 + (2.*sw_a2 + (3.*sw_a3 + (4.*sw_a4 + 5.*sw_a5*T)*T)*T)*T + \
        (sw_b1 + (2.*sw_b2 + (3.*sw_b3 + 4.*sw_b4*T)*T)*T)*S + \
        (sw_c1 + 2.*sw_c2*T)*S*(np.sqrt(S))
    alpha = - alpha / rho_ocean_ref
    return alpha

# Haline contraction coefficient
@jit(nopython=jit_no_python)
def sw_beta(T,S):
    beta = (sw_b0 + (sw_b1 + (sw_b2 + (sw_b3 + sw_b4*T)*T)*T)*T) + \
        1.5*(sw_c0 + (sw_c1 + sw_c2*T)*T)*np.sqrt(S) + 2.*sw_d0*S
    beta = beta / rho_ocean_ref
    return beta

#approximation of the liquidous temperature in Celsius. (only first term)
@jit(nopython=jit_no_python)
def liquidous(S):
    t_liq = -0.054*S
    return t_liq

@jit(nopython=jit_no_python)
def saturation_spec_hum(ts): #Bolton 1980
    ew = 6.112*np.exp(17.62*ts/(243.12+ts))
    ew = ew*100.
    ep = 0.62197  	# epsilon, ratio of molecular weight of water and dry air (0.622)
    sat_shum = (ep*ew)/(P_atm - (0.378*ew))
    #Using AOMIP definition of specific humidity. Can't find description of where 0.378 comes from
    return sat_shum


#same as used in Petty er al 2013
@jit(nopython=jit_no_python)
def saturation_spec_hum_petty(ts):
    return 157.6e6 / ((P_atm/1000. * np.exp(5420. / (ts+273.15))- 95.6e6))


@jit(nopython=jit_no_python)
def f(lat):
    f = 2*7.29e-5*np.sin(lat*np.pi/180.)
    return f


#################
##   Diffusion ##
#################
#YOu could spend a career playing with the values of diffusion in a 1d model to be told that you've wasted your career
@jit(nopython=jit_no_python)
def diffusion(temp,salt,oxy,u,v,kappa,kappa_u,nz,dt,dz):
    coef = dt/(dz**2)
    temp[1:nz-1] += kappa*(temp[0:nz-2]-2.*temp[1:nz-1]+temp[2:])*coef
    salt[1:nz-1] += kappa*(salt[0:nz-2]-2.*salt[1:nz-1]+salt[2:])*coef
    oxy[1:nz-1]  += kappa*(oxy[0:nz-2]-2.*oxy[1:nz-1]+oxy[2:])*coef
    u[1:nz-1]    += kappa_u*(u[0:nz-2]-2.*u[1:nz-1]+u[2:])*coef
    v[1:nz-1]    += kappa_u*(v[0:nz-2]-2.*v[1:nz-1]+v[2:])*coef
    return temp,salt,oxy,u,v


@jit(nopython=jit_no_python)
def advection_ekman(temp,salt,oxy,u,v,ekman,nz,dt,dz):
    coef = ekman*dt/(dz)
    temp[0:nz-1] -= (temp[0:nz-1]-temp[1:])*coef
    salt[0:nz-1] -= (salt[0:nz-1]-salt[1:])*coef
    oxy[0:nz-1]  -= (oxy[0:nz-1]-oxy[1:])*coef
    u[0:nz-1]    -= (u[0:nz-1]-u[1:])*coef
    v[0:nz-1]    -= (v[0:nz-1]-v[1:])*coef
    return temp,salt,oxy,u,v


#############################
##    Radiative Transfer   ##
#############################
@jit(nopython=jit_no_python)
def absorb(beta1,beta2,nz,dz):
#  Compute solar radiation absorption profile. This
#  subroutine assumes two wavelengths, and a double
#  exponential depth dependence for absorption.
#  Subscript 1 is for red, non-penetrating light, and
#  2 is for blue, penetrating light. rs1 is the fraction
#  assumed to be red.
    rs1 = 0.6
    rs2 = 1.0-rs1
    absorb = np.zeros(nz)
    z1 = np.arange(0,nz)*dz
    z2 = z1 + dz
    z1b1 = z1/beta1
    z2b1 = z2/beta1
    z1b2 = z1/beta2
    z2b2 = z2/beta2
    absorb = (rs1*(np.exp(-z1b1)-np.exp(-z2b1))+rs2*(np.exp(-z1b2)-np.exp(-z2b2)))
    return absorb


#############################
##  heat budget functions  ## T inputs in celsius
#############################

@jit(nopython=jit_no_python)
def lw_emission(T_in,emiss,stef_boltz=5.67e-8):
    return emiss*stef_boltz*((T_in+273.15)**4)

@jit(nopython=jit_no_python)
def lw_downwelling(lw,emiss):
    return emiss*lw

@jit(nopython=jit_no_python)
def sw_downwelling(sw,albedo): #think 1-IO here from petty saying this is all in their So so maybe remove
    return (1.-albedo)*sw

@jit(nopython=jit_no_python)
def ao_sens(T_so,T_a,U_a,cd_ocean):
    return rho_air_ref*cp_air*cd_ocean*U_a*(T_so-T_a)

@jit(nopython=jit_no_python)
def ao_latent(T_so,U_a,sat_sp_hum,sp_hum,cd_ocean):
    return rho_air_ref*Latent_vapor*cd_ocean*U_a*(sat_sp_hum-sp_hum)

#############################################
##   PWP Mixing Functions and Dependents   ##
#############################################

@jit(nopython=jit_no_python)
def mix_ts(t,s,d,ml_index):
    t[:ml_index+1] = np.mean(t[:ml_index+1])
    s[:ml_index+1] = np.mean(s[:ml_index+1])
    d[:ml_index+1] = density_0(t[0],s[0])
    return t,s,d

@jit(nopython=jit_no_python)
def mix_oxy(oxy,ml_index):
    oxy[:ml_index+1] = np.mean(oxy[0:ml_index+1])
    return oxy

@jit(nopython=jit_no_python)
def mix_uv(u,v,ml_index):
    u[:ml_index+1] = np.mean(u[:ml_index+1])
    v[:ml_index+1] = np.mean(v[:ml_index+1])
    return u,v

@jit(nopython=jit_no_python)
def remove_static_instabilities(ml_index,t,s): # if grid cell just below ML is lighter, mix into ML
    density = density_0(t,s)
    max_index = len(t)-2
    while density[ml_index] > density[ml_index+1] and ml_index<max_index:
        ml_index  += 1
        t,s,d = mix_ts(t,s,density,ml_index)
    return t,s,density,ml_index


@jit(nopython=jit_no_python)
def rot(ang,u,v):
    r_temp = (u+v*1j)*np.exp(ang*1j)
    u_temp = np.real(r_temp)
    v_temp = np.imag(r_temp)
    return u_temp,v_temp


@jit(nopython=jit_no_python)
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
    
    #denstemp=(d[j+1]-d[j])*temporary_f/2.
    #d[j+1] = d[j+1]-denstemp
    d[j:j+2] = density_0(t[j:j+2],s[j:j+2])
    
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


@jit(nopython=jit_no_python)
def grad_mix(dz,rg,nz,z,t,s,d,u,v,oxy,ml_index): #PWP mixing based on gradient richardson number

#  This function performs the gradeint Richardson Number relaxation
#  by mixing adjacent cells just enough to bring them to a new
#  Richardson Number.

#  Compute the gradeint Richardson Number, taking care to avoid dividing by
#  zero in the mixed layer.  The numerical values of the minimum allowable
#  density and velocity differences are entirely arbitrary, and should not
#  effect the calculations (except that on some occasions they evidnetly have!)
# the values I've used here for min dd/dv are arbitrary and appaear to recover solutions for where this function does not give errors
# imaginary density/negative salinity/looping forever.  one might put these in bulk richardson as well but no issue there yet
    rc 	= rg
    check=1
    j1 = 0
    j2 = nz-2
    r_check=np.zeros(nz)+9999999
    while check==1:
        for j in range(j1,j2):
            dd = (d[j+1]-d[j])/d[j]
            dv = (u[j+1]-u[j])**2+(v[j+1]-v[j])**2
            if dv < 1e-15: #changed this from ==0
                r_check[j] = 99999.
            elif dd<1e-15:
                r_check[j] = 99999
            else:
                r_check[j] = 9.81*dd*dz/dv
        min_rg = np.min(r_check)  #todo need index as well
        if (min_rg>rc):		# Check to see whether the smallest r is critical or not.
            check=0
        else:
    #  Mix the cells js and js+1 that had the smallest Richardson Number
            min_ind = int(np.argmin(r_check))
            js = min_ind
            t, s, d, u, v, oxy = stir(rc, min_rg, js, t, s, d, u, v, oxy)
            #  Recompute the Richardson Number over the part of the profile that has changed
            t, s, d, ml_index = remove_static_instabilities(ml_index, t, s)
            u, v = mix_uv(u,v,ml_index)
            oxy  = mix_oxy(oxy,ml_index)
            j1 = js-2
            if j1 < 0:
                j1 = 0
            j2 = js+2
            if (j2 > nz-2):
                j2 = nz-2
    return t, s, d, u, v, oxy, ml_index


@jit(nopython=jit_no_python)
def bulk_mix(ml_index,rb,d,u,v,t,s,oxy,z,nz): #mixing for PWP based on Bulk Richardson number arguements
    rvc = 0.65
    g	= 9.81		# gravitational acceleration
    for j in range(ml_index+1,int(nz)):
        h 	= z[j]
        dd 	= (d[j]-d[0])/d[0]
        dv 	= (u[j]-u[0])**2+(v[j]-v[0])**2
        if dv < 1E-15: #as above
            rv = 9999.
        else:
            rv = g*h*dd/dv #save rv below this in here and matlab and compare?
        if rv>rvc:
            break
        else:
            t, s, d= mix_ts(t,s,d,j)
            u, v   = mix_uv(u,v,j) 
            oxy    = mix_oxy(oxy,j)
            #ml_index = j
    return ml_index,d,u,v,t,s,oxy


#fake advection below ad_i/mixed layer. relax to initiail profile
@jit(nopython=jit_no_python)
def Ocean_relax(t,s,o,t_0,s_0,o_0,ad_i,OR_ts):
    t[ad_i:] -= OR_ts*(t[ad_i:]-t_0[ad_i:])
    s[ad_i:] -= OR_ts*(s[ad_i:]-s_0[ad_i:])
    o[ad_i:] -= OR_ts*(o[ad_i:]-o_0[ad_i:])
    return t, s, o


@jit(nopython=jit_no_python)
def Ocean_relax_ml(t,s,o,t_0,s_0,o_0,ad_i,mli,OR_ts):
    t[mli:ad_i] -= OR_ts*(t[mli:ad_i]-t_0[mli:ad_i])
    s[mli:ad_i] -= OR_ts*(s[mli:ad_i]-s_0[mli:ad_i])
    o[mli:ad_i] -= OR_ts*(o[mli:ad_i]-o_0[mli:ad_i])
    return t, s, o


#see louise Biddle-Clark Thesis for this
@jit(nopython=jit_no_python)
def Oxygen_change(t,s,oxy,d,UU,ml_depth,ml_index,dt,A,): #todo turn on ice

    oxy_sat = sw_sat_oxy(s[0],t[0])
    oxy_sat = oxy_sat*44.658
    oxy_sat = oxy_sat/d[0]
    oxy_sat = oxy_sat*1000.
    #Calculate the schmidt number -- using values and equation from Wanninkhof 2014
    Sc = 1920.4+(-135.6*t[0])+(5.2122*(t[0]**2))+(-0.10939*(t[0]**3))+(0.00093777*(t[0]**4))
    #Calculated gas exchange rate
    kk = (1.791e-5)*((UU*UU)*(Sc**(-0.5)))
    #ddoxy = kk*(oxy[0] - oxy_sat)*(dt/ml_depth)
    #change in oxy
    if A ==0: #if no sea ice is present
        ddoxy = kk*(oxy[0] - oxy_sat)*(dt/ml_depth)
    else:
        ddoxy = (kk*0.25)*(oxy[0]-oxy_sat)*(dt/ml_depth) # if sea ice is present - see Loose et al. 2014
    oxy[:ml_index+1] -= ddoxy
    return oxy

@jit(nopython=jit_no_python)
def sw_sat_oxy(S,T):
    # Eqn(4) of Weiss 1970
    T = (273.15 + T * 1.00024)/100.
    lnC = sw_sat_oxy_a1+sw_sat_oxy_a2/T+sw_sat_oxy_a3*np.log(T)+sw_sat_oxy_a4*T+S*(sw_sat_oxy_b1+sw_sat_oxy_b2*T+sw_sat_oxy_b3*(T**2))
    c = np.exp(lnC)
    return c

# below is modified sea ice similar to petty et al 2015
@jit(nopython=jit_no_python)
def aio_sens(T_si,T_a,U_a):
    return rho_air_ref*cp_air*C_turb_i*U_a*(T_si-T_a)

@jit(nopython=jit_no_python)
def aio_latent(U_a,sp_hum,sat_sp_hum_i):
    return rho_air_ref*Latent_sub*C_turb_i*U_a*(sat_sp_hum_i-sp_hum)


@jit(nopython=jit_no_python)
def ice_cond_heat(T_si, T_f, h_ice, h_snow):
    return cond_ice*cond_snow*(T_f-T_si)/(cond_ice*h_snow+cond_snow*h_ice)

@jit(nopython=jit_no_python)
def run_no_ice(temp,salt,A,h_i):
    mr      = 0.
    t_flux  = 0.
    sw_flux = 0.
    return temp,salt,mr,A,h_i,t_flux,sw_flux
    
    
@jit(nopython=jit_no_python)    
def run_bc_ice(temp,salt,dens0,ml_index,ml_depth,A,h_i,T_a,U_a,lw,sw,sp_hum,u_star_i,u_star_l):
    t_fp = liquidous(salt[0])
    if temp[0] < t_fp:
        mr = (dens0 * cp_ocean * (temp[0] - t_fp)) / 1000.0 / Latent_fusion
        temp[0:ml_index + 1] = t_fp
        salt[0:ml_index + 1] = salt[0:ml_index + 1] / (1 + A * mr)
        h_i -= mr * ml_depth
        A = A_grow
    elif h_i > 0.:
        mr = (dens0 * cp_ocean * (temp[0] - t_fp)) / 1000.0 / Latent_fusion
        temp[0:ml_index + 1] = t_fp
        salt[0:ml_index + 1] = salt[0:ml_index + 1] / (1 + (A * mr))
        h_i = h_i - mr * ml_depth
        A = A_melt
    else:
        mr = 0.
        A = 0.
        h_i = 0.
    t_flux = 0#mr*1000.0*Latent_fusion #in og this isnt fluxed to kt
    sw_flux=-salt[0]*A*mr
    return temp,salt,mr,A,h_i,t_flux,sw_flux


# solve for balance to find Delta T across ice
@jit(nopython=jit_no_python)    
def findroot(t_low, t_high,t_fp,h_snow,h_ice,T_a,U_a,lw,sw,sp_hum,cond_on,alb,emiss):
    t1 = t_high
    t0 = t_low
    t_diff = t1 - t0
    while (np.fabs(t_diff)>1E-4):
        f_diff = heat_fx(t1,t_fp,h_snow,h_ice,T_a,U_a,lw,sw,sp_hum,cond_on,alb,emiss) - \
            heat_fx(t0,t_fp,h_snow,h_ice,T_a,U_a,lw,sw,sp_hum,cond_on,alb,emiss)
        t2 = t1 - (heat_fx(t1,t_fp,h_snow,h_ice,T_a,U_a,lw,sw,sp_hum,cond_on,alb,emiss) * (t1 - t0) / f_diff)
        t0 = t1
        t1 = t2
        t_diff = t1 - t0
    return t1


@jit(nopython=jit_no_python)    
def heat_fx(t_so,t_fp,h_snow,h_ice,T_a,U_a,lw,sw,sp_hum,cond_on,alb,emiss):
    extflux = 0.
    if cond_on:
        extflux = ice_cond_heat(t_so, t_fp, h_ice, h_snow)
    sens = aio_sens(t_so, T_a, U_a)
    sat_sp_hum_i = saturation_spec_hum(t_so)
    lat = aio_latent(U_a, sp_hum, sat_sp_hum_i)
    olr =lw_emission(t_so,emiss)
    ilr = lw_downwelling(lw,emiss)
    isr = sw_downwelling(sw,alb)
    fx = sens + lat + olr - ilr - isr - extflux
    return fx

@jit(nopython=jit_no_python)    
def run_full_ice(temp,salt,dens0,ml_index,ml_depth,A,h_i,T_a,U_a,lw,sw,sp_hum,u_star_i,u_star_l,dt,Div_yr,h_snow=0.1):
    
    t_fp = liquidous(salt[0])
    cond_flux=0.
    if(h_snow<0.001):
        alb = ice_albedo
        emiss = ice_emiss
    else:
        alb = snow_albedo
        emiss = snow_emiss

    if h_i>0.0:
        T_si = findroot(temp[0]-20.,temp[0]+20.,t_fp,h_snow,h_i,T_a,U_a,lw,sw,sp_hum,cond_on,alb,emiss)
        if cond_on:
            cond_flux=ice_cond_heat(T_si,t_fp,h_i,h_snow)
    basal = (dens0 * cp_ocean * u_star_i * Stanton * (temp[0] - t_fp))
    ridge = 0.
    mr=(basal-cond_flux)/ rho_ice_ref / Latent_fusion
    if temp[0]<t_fp and h_i>0.0:
        if A<A_grow:
            latheat=(1.-A)*(dens0 * cp_ocean * u_star_l * (temp[0] - t_fp))
            dA = latheat/(Latent_fusion * rho_ice_ref * h_i)
            r_base=0.
            A-=dA*dt
        else:
            latheat=(1.-A)*(dens0 * cp_ocean * u_star_l * (temp[0] - t_fp))
            dA=0.
            r_base=0
            A=A_grow
            ridge=latheat*(1.-A)/(Latent_fusion*rho_ice_ref)
    elif h_i>0.0:
        if A > A_min:
            latheat = (1. - A) *(dens0 * cp_ocean * u_star_l * (temp[0] - t_fp))
            dA =  (1. - R_b) * latheat / (Latent_fusion * rho_ice_ref * h_i)
            A += -dA * dt
            r_base=(latheat*R_b)
            mr += (r_base)/rho_ice_ref/Latent_fusion
        else:
            latheat=0.
            dA=0.
            r_base=0.
            A=A_melt
            h_i=0
    h_i-= (mr+ridge)*dt

    if h_i<0.0:
        h_i=h_ice_min
        mr=0.
        basal=0.
        A=0.
        latheat=0.
        dA=0.
    sw_flux =(rho_ice_ref/rho_ocean_ref)*(salt[0]  - S_ice) * (A * (mr+ridge)+dA*h_i)
    t_flux = 1./(rho_ocean_ref*cp_ocean) * (A*basal+latheat)
    temp[0:ml_index + 1] += - t_flux* dt / ml_depth
    salt[0:ml_index + 1] += -sw_flux*dt/ml_depth

    #convert ice divergence to seconds
    Div = 0.
    if Div_yr:
        Div=dt/(Div_yr*365.*24.*60.*60.)
        
    if A>A_melt:
        A-=Div*A
        if A>A_grow:
            A=A_grow

    return temp,salt,mr,A,h_i,t_flux,sw_flux



