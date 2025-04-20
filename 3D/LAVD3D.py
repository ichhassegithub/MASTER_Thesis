#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:44:22 2025

@author: sarahvalent
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc
import matplotlib.pyplot as plt
from LAVD_setup_original_eth import *

nc_file = '/Users/sarahvalent/Downloads/toy_mountain_noncyclic_100m_3d.005.nc'
data = nc.Dataset(nc_file, 'r')

#data
u_old = data.variables['u'][:] 
v_old = data.variables['v'][:] 
w_old = data.variables['w'][:] 
time = data.variables['time'][:]
y = data.variables['y'][:]
x = data.variables['x'][:]
zu_3d = data.variables['zu_3d'][:]
zw_3d = data.variables['zw_3d'][:]
xu = data.variables['xu'][:] 
yv = data.variables['yv'][:] 


u, v, w = reorder_velocity(u_old, v_old, w_old)


#get rid of the mask on the data
import numpy.ma as ma
is_masked = ma.isMaskedArray(u)

if is_masked:
    u = u.filled(0)
nan_indices = np.where(np.isnan(u))
#print("NaN values u:", nan_indices)


is_masked = ma.isMaskedArray(v)
if is_masked:
    v = v.filled(0)
nan_indices = np.where(np.isnan(v))
#print("NaN values v:", nan_indices)

is_masked = ma.isMaskedArray(w)
if is_masked:
    w = w.filled(0)
nan_indices = np.where(np.isnan(w))
#print("NaN values w:", nan_indices)

periodic_x = False
periodic_y = False
periodic_z = False
periodic = [periodic_x, periodic_y, periodic_z]
bool_unsteady = False

# domain
xmin = x.min()+10
xmax = xu.max()-10
ymin = y.min()+10
ymax = yv.max()-10

topo_data = np.loadtxt("/Users/sarahvalent/Desktop/MASTER-THESIS/PYTHON/FTLE and LAVD/toymountain_topo100m.txt")

x_topo = np.linspace(x.min(), x.max(), topo_data.shape[1])  
y_topo = np.linspace(y.min(), y.max(), topo_data.shape[0]) 
topo_interpolator = RegularGridInterpolator((y_topo, x_topo), topo_data)  



X,Y,Z = np.meshgrid(x,y,zu_3d)
dx_data = X[0,1,0]-X[0,0,0] 
dy_data = Y[1,0,0]-Y[0,0,0] 
dz_data = Z[0,0,1]-Z[0,0,0] 

delta = [dx_data, dy_data, dx_data]
aux_grid = [delta[0]/2, delta[1]/2, delta[2]/2]


Ny = 150
Nx = 150

x_domain = np.linspace(xmin, xmax, Nx, endpoint = True) # array (Nx, )
y_domain = np.linspace(ymin, ymax, Ny, endpoint = True) # array (Ny, )

X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
Z_domain = topo_interpolator(np.c_[Y_domain.ravel(), X_domain.ravel()]).reshape(X_domain.shape)
z_offset = 100
Z_domain += z_offset

## interpolation

time_data = time.reshape(1,-1)

Interpolant = interpolant_unsteady(X, Y, Z, u, v, w, time_data)

Interpolant_u = Interpolant[0] 
Interpolant_v = Interpolant[1] 
Interpolant_w = Interpolant[2] 

# time 

t0 = time[0] 
tN = time[-1] 
dt = 0.1 
lenT = abs(tN-t0) 
#time = np.arange(t0, tN+dt, dt) # can be activated or original time is used

################
# trajectory plot
def Fmap_test(x0, y0, z0):
    
    X0 = np.array([x0, y0, z0]) # array (3, Nx*Ny*Nz)
    
    Fmap = integration_dFdt(time, X0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)[0] # array (Nt, 3, Nx*Ny*Nz)
    
    return Fmap

#initial values
x0_xy = X_domain.ravel()   
y0_xy = Y_domain.ravel() 
z0_xy = Z_domain.ravel() 
 
fmap = Fmap_test(x0_xy, y0_xy, z0_xy)

colors = ['b', 'g', 'r']
plt.figure(figsize=(10, 8))
for i in range(y_test.shape[1]):
        plt.plot(x_test[:, i], y_test[:, i],color=colors[i % 3])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectories of the initial conditions: "Spaghetti-plot"')
plt.show()


######
# LAVD 3D

def compute_omega(x0, y0, z0):
    
    X0 = np.array([x0, y0, z0]) # array (3, Nx*Ny*Nz)
    
    Fmap = integration_dFdt(time, X0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)[0] # array (Nt, 3, Nx*Ny*Nz) 
   
    omega = np.zeros(Fmap.shape)
    
    for i in range(Fmap.shape[0]): # time array
        #print(time[i])
        omega[i,:,:], V ,X0= vorticity(time[i], Fmap[i,:,:], X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, aux_grid)
    
    return omega,V,X0

omega_xy,V,X0 = compute_omega(x0_xy, y0_xy, z0_xy)


LAVD_xy = LAVD(omega_xy, time, omega_avg = None)

X0_xy = np.array(x0_xy).reshape(Ny, Nx)  # Array (Ny, Nx)
Y0_xy = np.array(y0_xy).reshape(Ny, Nx)  
Z0_xy = np.array(z0_xy).reshape(Ny, Nx)
LAVD_xy = np.array(LAVD_xy).reshape(Ny, Nx)
#print(LAVD_xy)

plt.figure(figsize=(8, 6))
contour = plt.contourf(x_domain, y_domain, LAVD_xy,levels = 256, cmap='inferno')
plt.colorbar(contour, label="LAVD Value")

plt.title("LAVD Field")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()


