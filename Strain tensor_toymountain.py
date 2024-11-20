#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:40:06 2024

@author: sarahvalent
 Eulerian rate-of-strain tensor for toy mountain 100m x 100m
"""


import numpy as np
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc
import matplotlib.pyplot as plt
from Setup_toymountain import *

# data of toymountain - 100m x 100m 
nc_file = '/Users/sarahvalent/Downloads/toy_mountain_noncyclic_100m_3d.005.nc'
data = nc.Dataset(nc_file, 'r')

#data
u_old = data.variables['u'][:] # u(t,z,x,y)
v_old = data.variables['v'][:] 
w_old = data.variables['w'][:] 
time = data.variables['time'][:]
y = data.variables['y'][:]
x = data.variables['x'][:]
zu_3d = data.variables['zu_3d'][:]
zw_3d = data.variables['zw_3d'][:]
xu = data.variables['xu'][:] 
yv = data.variables['yv'][:] 


u, v, w = reorder_velocity(u_old, v_old, w_old) #u(y,x,z,t)
# print("Reordered u shape:", u.shape)
# print("Reordered v shape:", v.shape)
# print("Reordered w shape:", w.shape) 
## eliminate the masked -- that is around the toy mountain
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

## boundary conditions: flow is not periodic and bool_unsteady = True, 
#means we include time in velociy data
periodic_x = False
periodic_y = False
periodic_z = False
periodic = [periodic_x, periodic_y, periodic_z]
bool_unsteady = False
## defining original meshgrid 
X,Y,Z = np.meshgrid(x,y,zu_3d)
dx_data = X[0,1,0]-X[0,0,0] 
dy_data = Y[1,0,0]-Y[0,0,0] 
dz_data = Z[0,0,1]-Z[0,0,0] 

## auxillary grid spacing
delta = [dx_data, dy_data, dx_data]
aux_grid = [delta[0]/2, delta[1]/2, delta[2]/2]

## defining the domain
xmin = x.min()+10
xmax = xu.max()-10
ymin = y.min()+10
ymax = yv.max()-10

# loading topography map and interpolating it 
topo_data = np.loadtxt("/Users/sarahvalent/Desktop/MASTER-THESIS/PYTHON/FTLE and LAVD/toymountain_topo100m.txt")

x_topo = np.linspace(x.min(), x.max(), topo_data.shape[1])  
y_topo = np.linspace(y.min(), y.max(), topo_data.shape[0]) 
topo_interpolator = RegularGridInterpolator((y_topo, x_topo), topo_data)  

Ny = 150
Nx = 150

x_domain = np.linspace(xmin, xmax, Nx, endpoint = True) # array (Nx, )
y_domain = np.linspace(ymin, ymax, Ny, endpoint = True) # array (Ny, )
X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
Z_domain = topo_interpolator(np.c_[Y_domain.ravel(), X_domain.ravel()]).reshape(X_domain.shape)
z_offset = 100
Z_domain += z_offset

dx = x_domain[1]-x_domain[0] 
dy = y_domain[1]-y_domain[0] 
dz_x, dz_y = np.gradient(Z_domain, dx, dy)
dz = np.mean(np.sqrt(dz_x**2 + dz_y**2))

Ny = X_domain.shape[0] 
Nx = X_domain.shape[1] 

# reshape time for interpolation
time_data = time.reshape(1,-1)

Interpolant = interpolant_unsteady(X, Y, Z, u, v, w, time_data)

Interpolant_u = Interpolant[0] 
Interpolant_v = Interpolant[1] 
Interpolant_w = Interpolant[2] 

## defining time spacing
t0 = time[0] 
tN = time[10]
dt = 0.1 #
time = np.arange(t0, tN+dt, dt) 
lenT = abs(tN-t0) #

## calculate Jacobian from the original velocities
jacobians = []  # shape (t, y, x, z, 3, 3)


for t in range(u.shape[3]):  
    u_t = u[:, :, :, t]
    v_t = v[:, :, :, t]
    w_t = w[:, :, :, t]
    
    dudx, dudy, dudz = np.gradient(u_t, dy, dx, dz, edge_order=2)
    dvdx, dvdy, dvdz = np.gradient(v_t, dy, dx, dz, edge_order=2)
    dwdx, dwdy, dwdz = np.gradient(w_t, dy, dx, dz, edge_order=2)
    
    
    jacobian_t = np.stack((
        np.stack((dudx, dudy, dudz), axis=-1),
        np.stack((dvdx, dvdy, dvdz), axis=-1),
        np.stack((dwdx, dwdy, dwdz), axis=-1)
    ), axis=-2)
    
    
    jacobians.append(jacobian_t)
jacobians = np.array(jacobians) 

#strain tensor
def strain_tensor(jacobians):
    S = 0.5 * (jacobians + np.transpose(jacobians, (0, 1, 2, 3, 5, 4)))  
    return S

S = strain_tensor(jacobians)
#print(dx, dy, dz)  

#max eigenvalue of S
eigen = np.linalg.eigvals(S) #shape (54, 100, 100, 82, 3) eigenvalue
max_eigen_z = np.max(eigen, axis=-2)  # shape (54, 100, 100, 3) max eigenvalue on z-axis
max_eigen = np.max(max_eigen_z, axis=-1)  # shape: (54, 100, 100) max eigenvalue of the 3 eigenvalues

t_index = 20  # point in time
max_eigen_slice = max_eigen[t_index, :, :]  # Shape: (100, 100)
x_domain = np.linspace(x.min(), x.max(), max_eigen_slice.shape[0])
y_domain = np.linspace(y.min(), y.max(), max_eigen_slice.shape[1])

plt.figure(figsize=(8, 6))
contour = plt.contourf(x_domain, y_domain, max_eigen_slice, levels=200, cmap='viridis')
plt.colorbar(contour, label="Max Eigenvalue")

plt.title(f"Max eigenvalue  at t={t_index}")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
