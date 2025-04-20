-
"""
@author: sarahvalent
function collection for the toymountain set up
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import svd

def reorder_velocity(u, v, w):
   
    u_reordered = np.transpose(u, (2, 3, 1, 0))  
    v_reordered = np.transpose(v, (2, 3, 1, 0))
    w_reordered = np.transpose(w, (2, 3, 1, 0))
  
    return u_reordered, v_reordered, w_reordered


def interpolant_unsteady(X, Y, Z, U, V, W, time, interpolation_method = 'linear'):
   
    Interpolant = []
    
    Interpolant.append(RegularGridInterpolator((Y[:,0,0], X[0,:,0], Z[0,0,:], time[0,:]), U,
                                               bounds_error = False, fill_value = 0, method = interpolation_method))
    Interpolant.append(RegularGridInterpolator((Y[:,0,0], X[0,:,0], Z[0,0,:], time[0,:]), V,
                                               bounds_error = False, fill_value = 0, method = interpolation_method))
    Interpolant.append(RegularGridInterpolator((Y[:,0,0], X[0,:,0], Z[0,0,:], time[0,:]), W,
                                               bounds_error = False, fill_value = 0, method = interpolation_method))               
        
    return Interpolant


def integration_dFdt(time, x, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady):
    
    # reshape x
    x = x.reshape(3, -1) # reshape array (3, Nx*Ny*Nz)
    
    # Initialize arrays for flow map and derivative of flow map
    Fmap = np.zeros((len(time), 3, x.shape[1])) # array (Nt, 3, Nx*Ny*Nz)
    dFdt = np.zeros((len(time)-1, 3, x.shape[1])) # array (Nt-1, 3, Nx*Ny*Nz)
    
    # Step-size
    dt = time[1]-time[0] # float
    
    counter = 0 # int

    # initial conditions
    Fmap[counter,:,:] = x
    
    # Runge Kutta 4th order integration with fixed step size dt
    for t in time[:-1]: 
        Fmap[counter+1,:, :], dFdt[counter,:,:] = RK4_step(t, Fmap[counter,:, :], dt, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)[:2]
        counter += 1
    
    return Fmap, dFdt

def RK4_step(t, x1, dt, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady):
    
    t0 = t # float
    
    # Compute x_prime at the beginning of the time-step by re-orienting and rescaling the vector field
    x_prime = velocity(t, x1, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady) # array(3, Nx*Ny*Nz)
    
    # compute derivative
    k1 = dt * x_prime # array(3, Nx*Ny*Nz)
    
    # Update position at the first midpoint.
    x2 = x1 + .5 * k1 # array(3, Nx*Ny*Nz)
     
    # Update time
    t = t0 + .5*dt # float
    
    # Compute x_prime at the first midpoint.
    x_prime = velocity(t, x2, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady) # array(3, Nx*Ny*Nz)
    
    # compute derivative
    k2 = dt * x_prime # array(3, Nx*Ny*Nz)

    # Update position at the second midpoint.
    x3 = x1 + .5 * k2 # array(3, Nx*Ny*Nz)
    
    # Update time
    t = t0 + .5*dt # float
    
    # Compute x_prime at the second midpoint.
    x_prime = velocity(t, x3, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady) # array(3, Nx*Ny*Nz)
    
    # compute derivative
    k3 = dt * x_prime # array(3, Nx*Ny*Nz)
    
    # Update position at the endpoint.
    x4 = x1 + k3 # array(3, Nx*Ny*Nz)
    
    # Update time
    t = t0+dt # float
    
    # Compute derivative at the end of the time-step.
    x_prime = velocity(t, x4, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady) # array(3, Nx*Ny*Nz)
    
    # compute derivative
    k4 = dt * x_prime
    
    # Compute RK4 derivative
    y_prime_update = 1.0 / 6.0*(k1 + 2 * k2 + 2 * k3 + k4) # array(3, Nx*Ny*Nz)
    
    # Integration y <-- y + y_primeupdate
    y_update = x1 + y_prime_update # array(3, Nx*Ny*Nz)
    
    return y_update, y_prime_update/dt

def velocity(t, x, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady):
   
    x_swap = np.zeros((x.shape[1], 4))
    x_swap[:,0] = x[1,:]
    x_swap[:,1] = x[0,:]
    x_swap[:,2] = x[2,:] 
    x_swap[:,3] = t
    
    u = Interpolant_u(x_swap)
    v = Interpolant_v(x_swap)
    w = Interpolant_w(x_swap)
    
    vel = np.array([u, v, w])
    
    return vel



def LAVD(omega, times, omega_avg = None):
    
    omega = omega.reshape((omega.shape[0], 3, -1))
    lenT = times[-1] - times[0] # calculate length of time interval
    dt = times[1] - times[0] # assume uniform dt
    print(lenT, dt)
    
    # Compute averaged vorticity if not specified in the args.
    if omega_avg is None:
        omega_avg = [] # list (Nt,)
        for t in range(omega.shape[0]):
            omega_avg.append(np.mean(omega[t,:,:], axis = -1))

    # Compute LAVD
    omega_avg = np.array(omega_avg)
    LAVD = np.zeros((omega.shape[2]))
    omega_dif = (omega[0,:,:].T - omega_avg[0,:]).T # 0 th step in the trapezoid rule
    LAVD += np.sqrt(omega_dif[0]**2 + omega_dif[1]**2 + omega_dif[2]**2) * dt / 2

    for t in tqdm(range(1,omega.shape[0]-1)): # integrate with the trapezoid rule
        omega_dif = (omega[t,:,:].T - omega_avg[t,:]).T
        LAVD += np.sqrt(omega_dif[0]**2+omega_dif[1]**2+omega_dif[2]**2) * dt
            
    indexN = omega.shape[0]-1
    omega_dif = (omega[indexN,:,:].T - omega_avg[indexN,:]).T # N th step in the trapezoid rule
    LAVD += np.sqrt(omega_dif[0]**2 + omega_dif[1]**2 + omega_dif[2]**2) * dt / 2
    
    return LAVD / lenT

def vorticity(t, x, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, aux_grid):
    
        
    x = x.reshape(3,-1)
    
    # define auxiliary grid spacing
    rho_x = aux_grid[0] # float
    rho_y = aux_grid[1] # float
    rho_z = aux_grid[2] # float
    
    X0, XL, XR, XU, XD, XF, XB = [], [], [], [], [], [], []
    
    for i in range(x.shape[1]):
        
        xr = x[0, i] + rho_x # float
        xl = x[0, i] - rho_x # float
         
        if periodic[0]:
            
            xr = (xr-X[0,0,0])%(X[0,-1,0]-X[0,0,0])+X[0,0,0] # float
            xl = (xl-X[0,0,0])%(X[0,-1,0]-X[0,0,0])+X[0,0,0] # float
        
        yu = x[1, i] + rho_y # float
        yd = x[1, i] - rho_y # float
        
        if periodic[1]:
            
            yu = (yu-Y[0,0,0])%(Y[-1,0,0]-Y[0,0,0])+Y[0,0,0] # float
            yd = (yd-Y[0,0,0])%(Y[-1,0,0]-Y[0,0,0])+Y[0,0,0] # float
            
        zf = x[2, i] + rho_z # float
        zb = x[2, i] - rho_z # float
        
        if periodic[2]:
            
            zf = (zf-Z[0,0,0])%(Z[0,0,-1]-Z[0,0,0])+Z[0,0,0] # float
            zb = (zb-Z[0,0,0])%(Z[0,0,-1]-Z[0,0,0])+Z[0,0,0] # float
        
        X0.append([x[0,i], x[1,i], x[2,i]])
        XL.append([xl, x[1,i], x[2,i]])
        XR.append([xr, x[1,i], x[2,i]])
        XU.append([x[0,i], yu, x[2,i]])
        XD.append([x[0,i], yd, x[2,i]])
        XF.append([x[0,i], x[1,i], zf])
        XB.append([x[0,i], x[1,i], zb])
    
    X0 = np.array(X0).transpose()
    XL = np.array(XL).transpose()
    XR = np.array(XR).transpose()
    XU = np.array(XU).transpose()
    XD = np.array(XD).transpose()
    XF = np.array(XF).transpose()
    XB = np.array(XB).transpose()
    
    # velocity on the auxiliary meshgrid
    vLend = velocity(t, XL, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)    
    vRend = velocity(t, XR, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)     
    vDend = velocity(t, XD, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)   
    vUend = velocity(t, XU, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)
    vFend = velocity(t, XF, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)   
    vBend = velocity(t, XB, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)
    
    omega = np.zeros((3, X0.shape[1]))*np.nan
    
    for i in range(X0.shape[1]):
        
         dwdy = (vUend[2,i]-vDend[2,i])/(2*rho_y)
         dvdz = (vFend[1,i]-vBend[1,i])/(2*rho_z)
         
         dudz = (vFend[0,i]-vBend[0,i])/(2*rho_z)
         dwdx = (vRend[2,i]-vLend[2,i])/(2*rho_x)
         
         dvdx = (vRend[1,i]-vLend[1,i])/(2*rho_x)
         dudy = (vUend[0,i]-vDend[0,i])/(2*rho_y)

         omega[:,i] = np.array([dwdy-dvdz, dudz-dwdx, dvdx-dudy])
                
    return omega


def gradient_flowmap(time, x, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, aux_grid, verbose = False):
    
    # define auxiliary grid spacing
    rho_x = aux_grid[0]
    rho_y = aux_grid[1]
    rho_z = aux_grid[2]
    
    XL, XR, XU, XD, XF, XB = [], [], [], [], [], []
    
    for i in range(x.shape[1]):
        
        xr = x[0, i] + rho_x # float
        xl = x[0, i] - rho_x # float
        yu = x[1, i] + rho_y # float
        yd = x[1, i] - rho_y # float
        zF = x[2, i] + rho_z # float
        zB = x[2, i] - rho_z # float
    
        XL.append([xl, x[1, i], x[2, i]])
        XR.append([xr, x[1, i], x[2, i]])
        XU.append([x[0, i], yu, x[2, i]])
        XD.append([x[0, i], yd, x[2, i]])
        XF.append([x[0, i], x[1, i], zF])
        XB.append([x[0, i], x[1, i], zB])
    
    XL = np.array(XL).transpose()
    XR = np.array(XR).transpose()
    XU = np.array(XU).transpose()
    XD = np.array(XD).transpose()
    XF = np.array(XF).transpose()
    XB = np.array(XB).transpose()
    
    # launch trajectories from auxiliary grid
    XLend = integration_dFdt(time, XL, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, verbose)[0] # array (Nt, 2)
    XRend = integration_dFdt(time, XR, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, verbose)[0] # array (Nt, 2)
    XDend = integration_dFdt(time, XD, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, verbose)[0] # array (Nt, 2)
    XUend = integration_dFdt(time, XU, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, verbose)[0] # array (Nt, 2)
    XFend = integration_dFdt(time, XF, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, verbose)[0] # array (Nt, 2)
    XBend = integration_dFdt(time, XB, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, verbose)[0] # array (Nt, 2)
    
   
    
    return iterate_gradient(XRend, XLend, XUend, XDend, XFend, XBend)

from numba import njit, prange
@njit(parallel = True)
def iterate_gradient(XRend, XLend, XUend, XDend, XFend, XBend):
    
    gradFmap = np.zeros((XLend.shape[0], 3, 3, XLend.shape[2])) # array (Nt, 3, 3, Nx*Ny)
    
    for i in prange(XLend.shape[2]):      # points 
            
        for j in prange(XLend.shape[0]): # time

            gradFmap[j,0,0,i] = (XRend[j,0,i]-XLend[j,0,i])/(XRend[0,0,i]-XLend[0,0,i])
            gradFmap[j,1,0,i] = (XRend[j,1,i]-XLend[j,1,i])/(XRend[0,0,i]-XLend[0,0,i])
            gradFmap[j,2,0,i] = (XRend[j,2,i]-XLend[j,2,i])/(XRend[0,0,i]-XLend[0,0,i])
        
            gradFmap[j,0,1,i] = (XUend[j,0,i]-XDend[j,0,i])/(XUend[0,1,i]-XDend[0,1,i])
            gradFmap[j,1,1,i] = (XUend[j,1,i]-XDend[j,1,i])/(XUend[0,1,i]-XDend[0,1,i])
            gradFmap[j,2,1,i] = (XUend[j,2,i]-XDend[j,2,i])/(XUend[0,1,i]-XDend[0,1,i])
                
            gradFmap[j,0,2,i] = (XFend[j,0,i]-XBend[j,0,i])/(XFend[0,2,i]-XBend[0,2,i])
            gradFmap[j,1,2,i] = (XFend[j,1,i]-XBend[j,1,i])/(XFend[0,2,i]-XBend[0,2,i])
            gradFmap[j,2,2,i] = (XFend[j,2,i]-XBend[j,2,i])/(XFend[0,2,i]-XBend[0,2,i])
            
      
    
            
    return gradFmap

def FTLE(gradFmap, lenT):
   
    # compute maximum singular value of deformation gradient
    sigma_max = SVD(gradFmap)[1][0,0] # float
    
    # If sigma_max < 1, then set to 1. This happens due to numerical inaccuracies or when the flow is compressible.
    # Since we inherently assumed that the flow is incompressible we set sigma_max = 1 if condition is violated.
    if sigma_max < 1:
        return 0
                        
    return 1/(lenT)*np.log(sigma_max) # float



def SVD(gradFmap):
   
    P, s, QT = svd(gradFmap)
    Q = QT.transpose()
    SIG = np.diag(s)
    #print(SIG[0,0])
    
    return P, SIG, Q

