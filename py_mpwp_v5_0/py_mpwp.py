# ctrl f "todo" for things todo#
# todo set scalar save to not be just at end (if troubleshooting want that data) just overwrite file?
# todo check switches
#
import numpy as np
import pandas as pd
import xarray as xr

import py_mpwp_functions as pypwp
import py_mpwp_iotools as io
from py_mpwp_default_params import params
import os
import sys

class py_mpwp_model():    
    
    #################################################
    ##   print()
    #################################################
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
        

        
    
    ################################################# some things here may not be needed with certain switches (mr) but
    ##   Initialize simulation paramaters/arrays   
    ## setting them to 0 here and setting the save means no other adjustment is needed
    ################################################# besides setting the switch. not huge cost and fewer ifs.
    def __init__(self,**kwargs):
        
        # load parameters
        globals().update(params)
        for name, value in kwargs.items():
            try:
                val = float(value)
                globals().update({name:val})
            except:
                globals().update({name:value})
        pypwp.init_eos(equation_of_state,alpha0,beta0)

        # Start log file
        print('Model py_mpwp v5.0', file=open(log_file, 'w'))
        print('Authors: K. Thielen and L. Biddle 2020', file=open(log_file, 'a'))
        print('Contributors: F. Roquet', file=open(log_file, 'a'))
        print(' ', file=open(log_file, 'a'))
        for name, value in params.items():
            print("Model parameter: {} = {}".format(name,globals()[name]), file=open(log_file, 'a'))
        print(' ', file=open(log_file, 'a'))
                    
        # init time
        self.iteration   = 0
        maxiter          = int(ndays*8.64e4/dt)
        self.time        = np.arange(0,maxiter+1)*dt/8.64e4 + init_day
        self.niter       = maxiter
        self.niter_save  = int(86400/dt*freq_save)     # save outputs every niter_save time steps
        if dt > 1./10.*2.*3.14/pypwp.f(lat):
            raise ValueError('Time step, dt, too large to accurately resolve the inertial period.')

        # init depth
        maxdepth    = dz * max_depth // dz
        nz          = int(maxdepth // dz) + 1
        z           = np.arange(0, nz) * dz

        # init state
        z, temp, salt, oxy, u, v = io.load_init_state(init_state_file,z,log_file)
        self.depth  = z
        self.nz     = len(z)
        self.T      = temp
        self.S      = salt
        self.O      = oxy
        self.u      = u
        self.v      = v

        # load forcings
        self.forcing= io.load_forcing_fields(forcing_file, self.time)
        
        #calculate ml depth of initial profile
        i=k=0
        density = pypwp.density_0(self.T, self.S)
        ddens = density - density[0]
        while k==0 and i<self.nz - 1 :
            i+=1
            if ddens[i]>0.03:
                k=i
        self.ml_index  = k
        self.ml_depth  = z[self.ml_index]
        
        # ice model: select mode
        if bc_ice:
            self.h_i       = h_i0
            self.A         = A_0
        elif full_ice:
            self.h_i       = h_i0
            self.A         = A_0
        else:
            self.h_i       = 0
            self.A         = 0

        # initialize radiative absorbtion profile
        beta1 		    = 0.6		# longwave extinction coefficient (0.6 m)
        beta2		    = 20. 		# shortwave extinction coefficient (20 m)
        self.absorb  = pypwp.absorb(beta1,beta2,self.nz,dz)
        
        #copy original profiles for the ocean relaxation scheme
        self.Trelax  = self.T
        self.Srelax  = self.S
        self.Orelax  = self.O

        self.output  = io.init_output(self.time,self.depth,self.niter_save)
        self.output = io.save_output(self.output,0,0, \
                                         self.ml_index,self.ml_depth,self.h_i,self.A,\
                                         temp,salt,oxy,density,\
                                         u,v,0.,0.,0.)
        
        print(' ', file=open(log_file, 'a'))
        print("iter,ml_depth,si_th,si_co,sst,sss,ml_index", file=open(log_file, 'a'))

        
    
    
    
    #################################################
    ##   Run the loop iterations
    #################################################
    def run(self):
        from tqdm import tqdm

        for kk in tqdm(range(self.niter),ncols=80):
            self.step()

        self.to_netcdf()
        

        


    #################################################
    ##   run one iteration
    #################################################
    def step(self):
                
        # start compute next iteration
        self.iteration += 1

        ##  Load state variables locally  ##
        z         = self.depth
        nz        = self.nz
        temp      = self.T
        salt      = self.S
        oxy       = self.O
        u         = self.u
        v         = self.v
        ml_index  = self.ml_index
        ml_depth  = z[ml_index]
        h_i       = self.h_i
        A         = self.A
        
        ##  Load Forcings data for time step  ##
        forcing   = self.forcing.iloc[self.iteration-1]
        lw        = forcing.lw
        sw        = forcing.sw
        T_a       = forcing.T_a
        U_a       = forcing.U_a
        sp_hum    = forcing.shum
        precip    = forcing.precip
        tx        = forcing.tx
        ty        = forcing.ty

        #print("%.2f" %  1,self.iteration,ml_depth,h_i,A,temp[0],salt[0],ml_index)
        ##  Relaxes to initial profile below certain depth (ad_i) - rudimentary 3d/2d paramaterization
        if ocean_relax_switch:
            OR_d           = (OR_days*24.*60.*60.)/dt
            OR_timescale   = 1./OR_d
            ad_i           = ml_index
            #ad_i		   = int(ad_depth/dz)  # use if you want a fixed depth below which relax
            temp,salt,oxy  = pypwp.Ocean_relax(temp, salt, oxy, \
                                              self.Trelax, self.Srelax, self.Orelax, ml_index,OR_timescale)
            # if wanting to do below a set depth (ad_i in params) - change ml_index here to ad_i

        ##  diffusion - at the moment t and s diffuse at same rate - copied from mPWP (Biddle-Clark)
        if diffusion_switch:
            temp,salt,oxy,u,v = pypwp.diffusion(temp,salt,oxy,u,v,kappa,kappa_u,nz,dt,dz)
            
        if ekman_switch:
            temp,salt,oxy,u,v = pypwp.advection_ekman(temp,salt,oxy,u,v,w_ekman,nz,dt,dz)

        ##  diffusion can change T/S values so recalc density
        density = pypwp.density_0(temp, salt)
        temp,salt,density = pypwp.mix_ts(temp,salt,density,ml_index)
        ml_depth=z[ml_index]
        
        #print("%.2f" %  2,self.iteration,ml_depth,h_i,A,temp[0],salt[0],ml_index)

        ############################
        ##    OCEAN HEAT BUDGET   ##
        ############################
        T_so = temp[0]
        ocean_albedo	= 0.06		#
        ocean_emiss	    = 0.97		#
        cd_air      = 0.0015 
        cd_ocean    = 0.001		# drag coefficient ocean
        cd_ice		= 0.0013	# drag coefficient of ice
        Latent_vapor    = 2.501e6


        ##  radiation terms
        ISW     = pypwp.sw_downwelling(sw, ocean_albedo)
        OLR 	= pypwp.lw_emission(T_so,ocean_emiss)
        ILR 	= pypwp.lw_downwelling(lw,ocean_emiss)
        ##  sensible and latent heat
        o_sens 	= pypwp.ao_sens(T_so,T_a,U_a,cd_ocean)
        sat_sp_hum 	= pypwp.saturation_spec_hum(T_so)
        o_lat 	= pypwp.ao_latent(T_so,U_a,sat_sp_hum,sp_hum,cd_ocean)
        ##  group terms (surface v penetrating)
        q_out	= OLR + o_sens + o_lat-ILR
        q_in	= ISW

        ##  buoyancy budget of Open ocean
        evap 	= o_lat/(1000.*Latent_vapor)
        emp 	= evap-precip #precip from forcing

        ##use snow albedo if there is snow, otherwise use ice props.
        #if h_snow>0.001:
        #    #si_emiss=snow_emiss
        #    si_albedo=snow_albedo
        #else:
        #    #si_emiss=ice_emiss
        #    si_albedo=ice_albedo
        Ice_sw = 0.  #*pypwp.sw_downwelling(sw, si_albedo)

        #######################################
        ##   CALCULATE FLUXES OF HEAT/SALT   ##
        #######################################

        #cd_air_ice=2.36*10**(-3)
        #tio=rho_air_ref*cd_air_ice*U_a**2/3.
        #u_star_i=(tio/rho_ocean_ref)**(1./2.)
        rho_air_ref	    = 1.275		# air density
        u_star_l = np.sqrt(cd_ocean * rho_air_ref / rho_ocean_ref) * U_a
        u_star_i = np.sqrt(cd_ice * rho_air_ref / rho_ocean_ref) * U_a

        ##  flux surface fluxes evenly across existing ML #ice here perfectly reflective
        Netsw=sum(((1.-A)*(q_in)+A*Ice_sw)*self.absorb[0:ml_index+1])
        NetOLR=(1.-A)*OLR
        NetILR = (1. - A) * ILR
        Neto_sens=(1.-A)*o_sens
        Neto_lat=(1.-A)*o_lat

        #print("%.2f" %  3,self.iteration,ml_depth,h_i,A,temp[0],salt[0],ml_index)
        cp_ocean	    = 4190.		#
        temp[:ml_index+1]    +=((1.-A)*(q_in)+A*Ice_sw)*self.absorb[0:ml_index+1]*dt/(dz*density[0:ml_index+1]*cp_ocean)
        temp[:ml_index+1]    = np.mean(temp[0:ml_index+1])
        density[:ml_index+1] = pypwp.density_0(temp[:ml_index+1], salt[:ml_index+1])
        temp[:ml_index+1]    -= ((1.-A)*q_out) * (dt / ( density[0] * cp_ocean * ml_depth))
        salt[:ml_index+1]    = salt[:ml_index+1]/(1.-(1.-A)*emp*dt/ml_depth)
        

        ##  Penetrating shortwave below ML depth and check stability
        temp[ml_index+1:] += (1.-A)*ISW*self.absorb[ml_index+1:]*dt/(dz*density[ml_index+1:]*cp_ocean)
        density = pypwp.density_0(temp, salt)
        temp, salt, density, ml_index = pypwp.remove_static_instabilities(ml_index, temp, salt)
        u, v = pypwp.mix_uv(u,v,ml_index)
        ml_depth = z[ml_index]
        
        #print("%.2f" %  4,self.iteration,ml_depth,h_i,A,temp[0],salt[0],ml_index)
        ##  run ice model
        if bc_ice:
            temp,salt,mr,A,h_i,t_flux,sw_flux = pypwp.run_bc_ice(temp,salt,density[0],  \
                                                    ml_index,ml_depth,A,h_i,T_a,U_a,    \
                                                    lw,sw,sp_hum,u_star_i,u_star_l)        
        elif full_ice:
            temp,salt,mr,A,h_i,t_flux,sw_flux = pypwp.run_full_ice(temp,salt,density[0],\
                                                    ml_index,ml_depth,A,h_i,T_a,U_a,    \
                                                    lw,sw,sp_hum,u_star_i,u_star_l,dt,Div_yr,h_snow)
        else:
            temp,salt,mr,A,h_i,t_flux,sw_flux = pypwp.run_no_ice(temp,salt,A,h_i)
        
        ##  make sure column still statically stable
        temp, salt, density, ml_index = pypwp.remove_static_instabilities(ml_index, temp, salt)
        ml_depth = z[ml_index]

        #print("%.2f" %  5,self.iteration,ml_depth,h_i,A,temp[0],salt[0],ml_index)
        
        #######################################################
        ##   Calculate fluxes and Kraus-Turner type mixing   ## this Kt is done as outlined in Biddle clark thesis
        #######################################################

        if kt_switch==1:
            
            grav = 9.81		# gravitational acceleration            
            dw = 10.		# Lemke and Manley 1984 scale depth of dissipation
            m_kt = 0.4		# Co-efficient for power provided by wind, in Kraus-Turner
            n_kt = 0.18		# Co-efficient for power provided by buoyancy, in K-T
            if 'equation_of_state'=='eos-80':
                alpha   = pypwp.sw_alpha(temp,salt)
                beta    = pypwp.sw_beta(temp,salt)
            else:
                alpha   = alpha0
                beta    = beta0
                
            fw_flux= -salt[0]*(((1.-A)*emp))+sw_flux
            #sol_flux = ((1. - A) * (q_out - q_in * 0.45) + A * Ice_sw) / (rho_ocean_ref * cp_ocean)
            sol_flux = ((1. - A) * (q_out) - (q_in * np.sum(self.absorb[0:ml_index+1]))) / (rho_ocean_ref * cp_ocean)
            #u_star_i=0.0
            temp_flux=sol_flux-t_flux
            # neglects ice shear - assume u_i=u_ocean (urel=0)
            #u_star = U_a*(((rho_air_ref/rho_ocean_ref)*cd_ocean))**(1./2.)		
            u_star = np.sqrt((A* u_star_i * u_star_i) + ((1 - A) * u_star_l * u_star_l))
            Pw = ((2.*m_kt)*np.exp(-ml_depth/dw)*u_star**3) 			# Power for mixing supplied by wind
            Bo = grav*(alpha*temp_flux - beta*fw_flux)	# buoyancy forcing
            Pb = (ml_depth/2.)*((1.+n_kt)*Bo-(1.-n_kt)*abs(Bo))	# Power for mixing supplied by buoyancy change?
            we = (Pw+Pb)/(ml_depth*grav*(alpha*(temp[0]-temp[ml_index+1])-beta*(salt[0]-salt[ml_index+1])))
            
        ###########################################################
        ##   Calculate mixed layer deepening from this balance   ##
        ###########################################################
            ml_max      = max_depth - dz  # max ml
            if we >= 0.:
                ml_depth_test = ml_depth + we * dt          #check motion due to ek over time step
                # if moves more than dz/2 increment and recalc balance
                while ml_depth_test > (ml_depth + (dz / 2.)) and ml_depth_test<ml_max: 
                    ml_index += 1
                    ml_depth = z[ml_index]
                    Pw = ((2.*m_kt)*np.exp(-ml_depth/dw)*u_star**3)   # Power for mixing supplied by wind
                    Pb = (ml_depth/2.)*((1.+n_kt)*Bo-(1.-n_kt)*abs(Bo))  # Power for mixing supplied by buoyancy change?
                    we = (Pw+Pb)/(ml_depth*grav*(alpha*(temp[0]-temp[ml_index+1])-beta*(salt[0]-salt[ml_index+1])))
                    ml_depth_test = ml_depth+we*dt
            else:
                #ml_depth_test = ml_depth + we * dt
                ml_depth_test = (Pw /(-Bo)) # sometimes this gives huge value when switching and results in artifacts.
                if ml_depth_test<ml_depth:
                    ml_depth=ml_depth_test
            if ml_depth < ml_min:
                ml_depth=ml_min
                ml_index = int(round(ml_depth / dz))
            elif ml_depth>ml_max:
                ml_depth=ml_max
                ml_index = int(round(ml_depth / dz))
            else:
                ml_index = int(round(ml_depth / dz))
                ml_depth = z[ml_index]

            temp, salt, density = pypwp.mix_ts(temp, salt, density, ml_index)
            u, v                = pypwp.mix_uv(u, v, ml_index)
            oxy                 = pypwp.mix_oxy(oxy, ml_index)

        else:
            i=k=0
            density = pypwp.density_0(temp, salt)
            ddens = density - density[0]
            while k==0 and i<nz - 1 :
                i+=1
                if ddens[i]>0.02:
                    k=i
            ml_index = max(k,2)
            ml_depth  = z[ml_index]

        ###########################
        ##   End Krauss-Turner   ##
        ###########################
        
        #print("%.2f" %  6,self.iteration,ml_depth,h_i,A,temp[0],salt[0],ml_index)
        
        
        ##   Start PWP u/v stuff does nothing if rb=rg=0 in params  ##
        ##  Time step the momentum equation.

        ##  Rotate the current throughout the water column
        ang = -pypwp.f(lat)*dt/2.	# angle for current rotation. angle equal to inertial rotation for
        u,v = pypwp.rot(ang,u,v)

        ##  Apply the wind stress to the mixed layer as it now exists.
        u[0:ml_index+1] += (tx/(ml_depth*density[0]))*dt
        v[0:ml_index+1] += (ty/(ml_depth*density[0]))*dt

        ## I've just commented this out. uconn is not set in mpwp or pwp found online
        ## think this is antiquated section
        #  Apply drag to the current (this is a horrible parameterization of
        #  inertial-internal wave dispersion).

        #if ucon > 1E-10:
        #    u = u*(1-dt*ucon)
        #    v = v*(1-dt*ucon)

        ##  Rotate another half time step.
        u,v = pypwp.rot(ang,u,v)

        ##  Finished with the momentum equation for this time step.
        ##  Do the bulk Richardson number instability form of mixing (as in PWP).

        if rb > 0: # Switch for bulk richardson
            ml_index,density,u,v,temp,salt,oxy = pypwp.bulk_mix(ml_index,rb,density,u,v,temp,salt,oxy,z,nz)
            ml_depth = z[ml_index]
        #print("%.2f" %  7,self.iteration,ml_depth,h_i,A,temp[0],salt[0],ml_index)

        if rg > 0: # Switch for gradient richardson
            temp,salt,density,u,v,oxy,ml_index = pypwp.grad_mix(dz,rg,nz,z,temp,salt,density,u,v,oxy,ml_index)
            ml_depth = z[ml_index]

        oxy = pypwp.Oxygen_change(temp, salt, oxy, density, U_a, ml_depth,ml_index, dt,A)

        #print("%.2f" %  8,self.iteration,ml_depth,h_i,A,temp[0],salt[0],ml_index)
                
        if self.iteration%self.niter_save==0:
            k = int(self.iteration/self.niter_save)
            print("%.2f" %  self.iteration,ml_depth,h_i,A,temp[0],salt[0],ml_index, file=open(log_file, 'a'))
            self.output = io.save_output(self.output,k,self.iteration, \
                                         ml_index,ml_depth,h_i,A,temp,salt,oxy,density,\
                                         u,v,sw_flux,t_flux,u_star_l)

        ##  Load state variables locally  ##
        self.T    = temp 
        self.S    = salt 
        self.O    = oxy 
        self.u    = u 
        self.v    = v 
        self.ml_index = ml_index
        self.ml_depth = z[ml_index]
        self.h_i  = h_i 
        self.A    = A 
        
        
    ###############################################
    ##                   save outputs            ##
    ###############################################
    def to_netcdf(self):
        fname = output_file
        if fname[-3:]!='.nc':
            fname = fname+'.nc'
        self.output.to_netcdf(fname,mode='w')

    

###########################################################
##          End of py_mpwp_model class definition        ##
###########################################################



# main
if __name__ == "__main__":
    # execute only if run as a script
    from timeit import default_timer as timer
    import argparse
    
    global output_file, init_state_file, forcing_file, log_file

    parser = argparse.ArgumentParser(description='py_mpwp model, v4.0.')
    parser.add_argument("--output_file",dest="output_file",default='output.nc',
                          help="name of output file (.nc format)")
    parser.add_argument("--init_state_file",dest="init_state_file",default="init_state.npz",
                          help="name of initial profile file (.nc, .npz, .npz)")
    parser.add_argument("--forcing_file",dest="forcing_file",default="forcing.npz",
                          help="name of forcing file (.nc, .npz, .mat)")
    parser.add_argument("--log_file",dest="log_file",default='output.txt',
                          help="name of log file (.txt format)")
    args = vars(parser.parse_args())
    
    output_file          = args['output_file']
    log_file             = args['log_file']
    forcing_file         = args['forcing_file']
    init_state_file      = args['init_state_file']

    start = timer()
    my_model = py_mpwp_model()
    my_model.run()
    end = timer()
    print('Succes. Running time [s]: ',end - start)
    

