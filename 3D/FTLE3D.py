#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:40:19 2024

@author: sarahvalent
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc
import matplotlib.pyplot as plt
from LAVD_setup_original_eth import *



#nc_file = '/Users/sarahvalent/Desktop/MASTER-THESIS/SHARED FOLDER/toy_mountain_noncyclic_100m_3d.001.nc'
nc_file = '/Users/sarahvalent/Downloads/toy_mountain_noncyclic_100m_3d.005.nc'

#nc_file = '/Users/sarahvalent/Desktop/MASTER-THESIS/toy_mountain_smooth_inflow_3d.001.nc'
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
#print(time)



u, v, w = reorder_velocity(u_old, v_old, w_old)
#time = time[np.newaxis, :]  # time shape (1,100)

print("Reordered u shape:", u.shape)
print("Reordered v shape:", v.shape)
print("Reordered w shape:", w.shape) #(100, 100,100,22)



#set boundaries

# t0 = time.min() 
# tN = time.max()
# #dt = 0.01*np.sign(tN-t0) 
# lenT = abs(tN-t0)




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

xmin = x.min()+10
xmax = xu.max()-10
ymin = y.min()+10
ymax = yv.max()-10
#zmin = 500
#zmax = 500

topo_data = np.loadtxt("/Users/sarahvalent/Desktop/MASTER-THESIS/PYTHON/FTLE and LAVD/toymountain_topo100m.txt")

x_topo = np.linspace(x.min(), x.max(), topo_data.shape[1])  
y_topo = np.linspace(y.min(), y.max(), topo_data.shape[0]) 
topo_interpolator = RegularGridInterpolator((y_topo, x_topo), topo_data)  



X,Y,Z = np.meshgrid(x,y,zu_3d)
dx_data = X[0,1,0]-X[0,0,0] 
dy_data = Y[1,0,0]-Y[0,0,0] 
dz_data = Z[0,0,1]-Z[0,0,0] 

delta = [dx_data, dy_data, dx_data]
aux_grid = [0.01,0.01,0.01]




Ny = 150
Nx = 150
Nz = 150

x_domain = np.linspace(xmin, xmax, Nx, endpoint = True) # array (Nx, )
y_domain = np.linspace(ymin, ymax, Ny, endpoint = True) # array (Ny, )
#z_domain = np.linspace(zmin, zmax, Nz, endpoint = True) # array (Nz, )

X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
Z_domain = topo_interpolator(np.c_[Y_domain.ravel(), X_domain.ravel()]).reshape(X_domain.shape)
z_offset = 100
Z_domain += z_offset


# dx = x_domain[1]-x_domain[0] # float
# dy = y_domain[1]-y_domain[0] # float
# #dz = z_domain[1]-z_domain[0] # float
# dz_x, dz_y = np.gradient(Z_domain, dx, dy)
# dz = np.mean(np.sqrt(dz_x**2 + dz_y**2))

#X_domain, Y_domain, Z_domain = np.meshgrid(x_domain, y_domain, z_domain) 

Ny = X_domain.shape[0] # int
Nx = X_domain.shape[1] # int
#Nz = X_domain.shape[2] # int

#time = np.linspace(0, 1, 100)

time_data = time.reshape(1,-1)

Interpolant = interpolant_unsteady(X, Y, Z, u, v, w, time_data)

Interpolant_u = Interpolant[0] 
Interpolant_v = Interpolant[1] 
Interpolant_w = Interpolant[2] 

# Interpolant_u = RegularGridInterpolator((y,x, zu_3d, time),u, method = 'linear', bounds_error = False, fill_value = np.nan)
# Interpolant_v = RegularGridInterpolator((y,x, zu_3d, time),v, method = 'linear', bounds_error = False, fill_value = np.nan)
# Interpolant_w = RegularGridInterpolator((y,x, zw_3d, time),w, method = 'linear', bounds_error = False, fill_value = np.nan)


# x_val = 5275      
# y_val = 4272      
# z_val = 500       
# t_val = time[0]      


# u_interp_val = Interpolant_u([y_val, x_val, z_val, t_val])
# v_interp_val = Interpolant_v([y_val, x_val, z_val, t_val])
# w_interp_val = Interpolant_w([y_val, x_val, z_val, t_val])


# print("Interpolated U value:", u_interp_val)
# print("Interpolated V value:", v_interp_val)
# print("Interpolated W value:", w_interp_val)


# t_index = 5
# y_index = 5
# x_index = 5
# z_index = 5

# x_index = 5275     
# y_index = 4272     
# z_index = 500      
# t_index = 10  

# y_test = Y[y_index, 0, 0]    
# x_test = X[0, x_index, 0]    
# z_test = Z[0, 0, z_index]    
# t_test = time[ t_index]    

# test_point = [y_test, x_test, z_test, t_test]


# u_interp_val = Interpolant_u(test_point)
# v_interp_val = Interpolant_v(test_point)
# w_interp_val = Interpolant_w(test_point)


# u_true = u[y_index, x_index, y_index, z_index]
# v_true = v[y_index, x_index, y_index, z_index]
# w_true = w[y_index, x_index, y_index, z_index]


# print("True U value:", u_true)
# print("Interpolated U value:", u_interp_val)
# print("True V value:", v_true)
# print("Interpolated V value:", v_interp_val)
# print("True W value:", w_true)
# print("Interpolated W value:", w_interp_val)


t0 = time[0] 
tN = time[-1]
# dt = 0.1 #
# time = np.arange(t0, tN+dt, dt) 
lenT = abs(tN-t0) #




def compute_FTLE(x0, y0, z0):
    
    X0 = np.array([x0, y0, z0]) # array (3, Nx*Ny*Nz)
    
    DF, test_iteration = gradient_flowmap(time, X0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, aux_grid) # array (Nt, 3, 3, Nx*Ny*Nz)
 
    ftle = []
    
    # iterate over initial conditions/points
    for i in range(DF.shape[3]): 
        ftle.append(FTLE(DF[-1,:,:,i], lenT)) # for each point the last time step!
    
    return ftle, test_iteration,DF




x0_xy = X_domain.ravel()   # (Nx*Ny,)
y0_xy = Y_domain.ravel() 
z0_xy = Z_domain.ravel()

omega_xy, test_iteration,DF = compute_FTLE(x0_xy, y0_xy, z0_xy)


# X0_xy = np.array(x0_xy).reshape(Ny, Nx)  # Array (Ny, Nx)
# Y0_xy = np.array(y0_xy).reshape(Ny, Nx)  
# Z0_xy = np.array(z0_xy).reshape(Ny, Nx)
omega_xy = np.array(omega_xy).reshape(Ny, Nx)

plt.figure(figsize=(8, 6))
contour = plt.contourf(x_domain, y_domain, omega_xy, levels=200, cmap='viridis')

cbar = plt.colorbar(contour, label="FTLE Value")
cbar.ax.tick_params(labelsize=16)  
cbar.set_label("FTLE Value", fontsize=14)  
plt.title("FTLE Field - turbulent recycling", fontsize=16)
plt.xlabel("X-axis", fontsize=16)
plt.ylabel("Y-axis", fontsize=16)
plt.show()



#print(test_iteration.shape)

# XRend = test_iteration[0]
# XLend = test_iteration[1]
# XUend = test_iteration[2]
# XDend = test_iteration[3]
# XFend = test_iteration[4]
# XBend = test_iteration[5]




 
# y_test = XRend[:, 1, :]  
# x_test = XRend[:, 0, :] 
# z_test = XRend[: ,2, :]


# plt.plot(x_test[:, 1291], z_test[:, 1291]) 
# plt.show()

# plt.figure(figsize=(10, 8))
# for i in range(y_test.shape[1]):
#         plt.plot(x_test[:, i], y_test[:, i])  
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()


# trajectory_index = 113

# x_test = XLend[:, 0, trajectory_index]  
# y_test = XLend[:, 1, trajectory_index]  


# plt.figure(figsize=(8, 6))
# plt.plot(x_test, y_test, 'b-', label=f'Trajectory {trajectory_index}')
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.title("Trajectory for Selected Initial Condition")
# plt.legend()
# plt.grid(True)
# plt.axis("equal")
# plt.show()

# y_test = XDend[:, 1, :]  
# x_test = XDend[:, 0, :] 

# y_test_aligned = np.zeros_like(y_test)
# x_test_aligned = np.zeros_like(x_test)





# backward_indices = []

# for i in range(y_test.shape[1]):
#     initial_y = y_test[0, i]  # Time step = 1
#     y_test_aligned[:, i] = 1 + y_test[:, i] - initial_y

#     initial_x = x_test[0, i]  # Time step = 0
#     x_test_aligned[:, i] = x_test[:, i] - initial_x

#     if x_test[1, i] < x_test[0, i]:
#         backward_indices.append(i)  
#         print(f"Trajectory {i + 1} ")
#         print(f"Initial condition X0, Y0: ({x_test[0, i]}, {y_test[0, i]})")


# plt.figure(figsize=(10, 8))
# for i in range(y_test_aligned.shape[1]):
#     if i not in backward_indices:  
#         plt.plot(x_test_aligned[:, i], y_test_aligned[:, i])

# plt.xlabel("X, Starting Point X = 0")
# plt.ylabel("Y; Starting Point Y = 1")
# plt.title("Aligned Trajectories (Excluding Backward Movements)")
# plt.grid(True)

# plt.show()

# plt.figure(figsize=(10, 8))
# for i in backward_indices:  # Loop through the identified backward indices
#     plt.plot(x_test_aligned[:, i], y_test_aligned[:, i], label=f'Backward Trajectory {i + 1}')

# plt.xlabel("X, Starting Point X = 0")
# plt.ylabel("Y; Starting Point Y = 1")
# plt.title("Aligned Backward Trajectories")
# plt.grid(True)
# plt.legend()  # Optional: show legend if desired
# plt.show()



