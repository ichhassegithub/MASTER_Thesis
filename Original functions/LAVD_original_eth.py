#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:37:56 2024

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



def reorder_velocity(u, v, w):
    """
    Parameters:
    u (ndarray): u vector with shape (time, z, y, x).
    v (ndarray): v vector with shape (time, z, y, x).
    w (ndarray): w vector with shape (time, z, y, x).

    Returns:
    tuple: Reordered u, v, w vectors with shape (y, x, z, t).
    """

    # Transpose the arrays to reorder them from (time, z, y, x) to (y, x, z, time)
    u_reordered = np.transpose(u, (2, 3, 1, 0))  # Change axes to (y, x, z, t)
    v_reordered = np.transpose(v, (2, 3, 1, 0))
    w_reordered = np.transpose(w, (2, 3, 1, 0))
    
    # u_reordered = np.transpose(u, (3, 2, 1, 0))  # Change axes to (x, y, z, t)
    # v_reordered = np.transpose(v, (3, 2, 1, 0))
    # w_reordered = np.transpose(w, (3, 2, 1, 0))
    
    return u_reordered, v_reordered, w_reordered

u, v, w = reorder_velocity(u_old, v_old, w_old)

# print("Reordered u shape:", u.shape)
# print("Reordered v shape:", v.shape)
# print("Reordered w shape:", w.shape) #(100, 100,100,22)



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
aux_grid = [delta[0]/2, delta[1]/2, delta[2]/2]




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


time_data = time.reshape(1,-1)

Interpolant = interpolant_unsteady(X, Y, Z, u, v, w, time_data)

Interpolant_u = Interpolant[0] 
Interpolant_v = Interpolant[1] 
Interpolant_w = Interpolant[2] 



t0 = time[0] # float

# Final time
tN = time[-1] # float

# Time step-size
dt = 0.1 # float

time = np.arange(t0, tN+dt, dt) # shape (Nt,)

# Length of time interval
lenT = abs(tN-t0) #


def Fmap_test(x0, y0, z0):
    
    X0 = np.array([x0, y0, z0]) # array (3, Nx*Ny*Nz)
    
    Fmap = integration_dFdt(time, X0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)[0] # array (Nt, 3, Nx*Ny*Nz)
    
    return Fmap
########### without batches!
# xy
x0_xy = X_domain.ravel()   # (Nx*Ny,)
y0_xy = Y_domain.ravel() 
z0_xy = Z_domain.ravel()  # only one value
fmap = Fmap_test(x0_xy, y0_xy, z0_xy)
#print(fmap.shape)# (nt,3,Nx*Ny*Nz)
#print(test[-1].reshape(Nx,Ny,Nz))

### one trajectory

# trajectory_index = 113

# # Extract x and y components for this trajectory across all timesteps
# x_test = test[:, 0, trajectory_index]  # x-coordinates over time
# y_test = test[:, 1, trajectory_index]  # y-coordinates over time

# # Plotting the single trajectory
# plt.figure(figsize=(8, 6))
# plt.plot(x_test, y_test, 'bo', label=f'Trajectory {trajectory_index}')
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.title("Trajectory for Selected Initial Condition")
# plt.legend()
# plt.grid(True)
# plt.axis("equal")
# plt.show()



y_test = fmap[:, 1, :]  
x_test = fmap[:, 0, :] 
z_test = fmap[: ,2, :]


# plt.plot(x_test[:, 1291], z_test[:, 1291]) 
# plt.show()
colors = ['b', 'g', 'r']
plt.figure(figsize=(10, 8))
for i in range(y_test.shape[1]):
        plt.plot(x_test[:, i], y_test[:, i],color=colors[i % 3])
plt.xlabel('X', fontsize = 14)
plt.ylabel('Y', fontsize = 14)
plt.title('Trajectories of turbulent inflow', fontsize = 14)
plt.show()




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
#     plt.plot(x_test_aligned[:, i], y_test_aligned[:, i])#, label=f'Backward Trajectory {i + 1}')

# plt.xlabel("X, Starting Point X = 0")
# plt.ylabel("Y; Starting Point Y = 1")
# plt.title("Aligned Backward Trajectories")
# plt.grid(True)
# plt.legend()  # Optional: show legend if desired

#plt.show(x_test_aligned[:, backward_indices[1]],z_test[:,backward_indices[1]] )


#plt.plot()








def compute_omega(x0, y0, z0):
    
    X0 = np.array([x0, y0, z0]) # array (3, Nx*Ny*Nz)
    
    Fmap = integration_dFdt(time, X0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)[0] # array (Nt, 3, Nx*Ny*Nz) 
   
    omega = np.zeros(Fmap.shape)
    
    for i in range(Fmap.shape[0]): # time array
        #print(time[i])
        omega[i,:,:], V ,X0= vorticity(time[i], Fmap[i,:,:], X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, aux_grid)
    
    return omega,V,X0



# x0_xy = X_domain[:, :, 1].ravel()   # (Nx*Ny,)
# y0_xy = Y_domain[:, :, 1].ravel() 
# z0_xy = Z_domain[:, :, 1].ravel() 

x0_xy = X_domain.ravel()   # (Nx*Ny,)
y0_xy = Y_domain.ravel() 
z0_xy = Z_domain.ravel()


omega_xy,V,X0 = compute_omega(x0_xy, y0_xy, z0_xy)


LAVD_xy = LAVD(omega_xy, time, omega_avg = None)

X0_xy = np.array(x0_xy).reshape(Ny, Nx)  # Array (Ny, Nx)
Y0_xy = np.array(y0_xy).reshape(Ny, Nx)  
Z0_xy = np.array(z0_xy).reshape(Ny, Nx)
LAVD_xy = np.array(LAVD_xy).reshape(Ny, Nx)
#print(LAVD_xy)

plt.figure(figsize=(8, 6))

contour = plt.contourf(x_domain, y_domain, LAVD_xy, levels=256, cmap='inferno')

cbar = plt.colorbar(contour, label="LAVD Value")
cbar.ax.tick_params(labelsize=16)  
cbar.set_label("LAVD Value", fontsize=16)  
plt.title("LAVD Field", fontsize=16)
plt.xlabel("X-axis", fontsize=16)
plt.ylabel("Y-axis", fontsize=16)

plt.show()

# XRend = V[0]
# XLend = V[1]
# XUend = V[2]
# XDend = V[3]
# XFend = V[4]
# XBend = V[5]




################################################


# def vort(x0,y0,z0):
#     X0 = np.array([x0, y0, z0])#
#     vort_test = np.zeros(test.shape)
#     for i in range(len(time)):
#         vort_test[i,:,:],A,B= vorticity(time[i], X0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, aux_grid)
#     #vort_test[time[0],:,:],A,B= vorticity(time[0], X0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady, aux_grid)
    
#     return   vort_test

# #vorticity_test = vort(x0_xy, y0_xy, z0_xy)
# def vort_single(x0, y0, z0, t_index=0):
#     X0 = np.array([[x0], [y0], [z0]])
#     vorticity_at_point, _, _ = vorticity(time[t_index], X0, X, Y, Z, 
#                                           Interpolant_u, Interpolant_v, Interpolant_w, 
#                                         periodic, bool_unsteady, aux_grid)
    
#     return vorticity_at_point,X0


# #vort_at_point_single = vort_single(60, 60 ,500)
# #vort_at_point,X0_ = vort_single(x0_xy, y0_xy, z0_xy)
# #X0_ = X0_.reshape(3,-1)



# def compute_vorticity(time, x0, y0, z0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, aux_grid):


#     X0 = np.array([x0, y0, z0]).reshape(3, -1)
#     N_points = X0.shape[1]
#     rho_x, rho_y, rho_z = aux_grid
#     vorticity_values = np.zeros((3, N_points))  
#     for i in range(N_points):
     
#         x, y, z = X0[:, i]
        
       
#         xr, xl = x + rho_x, x - rho_x
#         yu, yd = y + rho_y, y - rho_y
#         zf, zb = z + rho_z, z - rho_z
        
#         uL = Interpolant_u([yd, xl, z, time])[0]
#         uR = Interpolant_u([y, xr, z, time])[0]
#         vD = Interpolant_v([yd, x, z, time])[0]
#         vU = Interpolant_v([yu, x, z, time])[0]
#         wF = Interpolant_w([y, x, zf, time])[0]
#         wB = Interpolant_w([y, x, zb, time])[0]

#         dwdy = (wF - wB) / (2 * rho_y)
#         dvdz = (vU - vD) / (2 * rho_z)
        
#         dudz = (uR - uL) / (2 * rho_z)
#         dwdx = (uL - uR) / (2 * rho_x)
        
#         dvdx = (vU - vD) / (2 * rho_x)
#         dudy = (uR - uL) / (2 * rho_y)

#         vorticity_values[:, i] = np.array([dwdy - dvdz, dudz - dwdx, dvdx - dudy])

#     return vorticity_values



# #o = compute_vorticity(time[0], x0_xy, y0_xy, z0_xy, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, aux_grid)



# def vort_test_compute(x0, y0, z0):
    
#     X0 = np.array([x0, y0, z0]) # array (3, Nx*Ny*Nz)
    
#     Fmap = integration_dFdt(time, X0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)[0] # array (Nt, 3, Nx*Ny*Nz) 
   
#     omega = np.zeros(Fmap.shape)
    
#     for i in range(Fmap.shape[0]): # time array
#         #print(time[i])
#         x_t, y_t, z_t = Fmap[i, 0, :], Fmap[i, 1, :], Fmap[i, 2, :]
        
#         # Step 5: Compute vorticity for the current timestep using the current positions
#         omega[i, :, :] = compute_vorticity(time[i], x_t, y_t, z_t, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, aux_grid)
    
#     return omega
# #o_test = vort_test_compute(x0_xy, y0_xy, z0_xy)







# # LAVD_xy = LAVD(vorticity_test, time, omega_avg = None)

# # X0_xy = np.array(x0_xy).reshape(Ny, Nx)  # Array (Ny, Nx)
# # Y0_xy = np.array(y0_xy).reshape(Ny, Nx)  
# # Z0_xy = np.array(z0_xy).reshape(Ny, Nx)
# # LAVD_xy = np.array(LAVD_xy).reshape(Ny, Nx)
# # #print(LAVD_xy)

# # plt.figure(figsize=(8, 6))
# # contour = plt.contourf(x_domain, y_domain, LAVD_xy,levels = 256, cmap='inferno')
# # plt.colorbar(contour, label="LAVD Value")

# # plt.title("LAVD Field")
# # plt.xlabel("X-axis")
# # plt.ylabel("Y-axis")

# # plt.show()
