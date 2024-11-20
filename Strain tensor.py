#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:40:06 2024

@author: sarahvalent
"""


import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter
from scipy.linalg import eigh
import netCDF4 as nc
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from LAVD_setup_original_eth import *
from tqdm import tqdm
from plotly import graph_objs as go 

#nc_file = '/Users/sarahvalent/Desktop/MASTER-THESIS/SHARED FOLDER/toy_mountain_noncyclic_100m_3d.001.nc'
# nc_file = '/Users/sarahvalent/Downloads/toy_mountain_noncyclic_100m_3d.005.nc'
# data = nc.Dataset(nc_file, 'r')

# #data
# u_old = data.variables['u'][:] 
# v_old = data.variables['v'][:] 
# w_old = data.variables['w'][:] 
# time = data.variables['time'][:]
# y = data.variables['y'][:]
# x = data.variables['x'][:]
# zu_3d = data.variables['zu_3d'][:]
# zw_3d = data.variables['zw_3d'][:]
# xu = data.variables['xu'][:] 
# yv = data.variables['yv'][:] 



# def reorder_velocity(u, v, w):
#     """
#     Parameters:
#     u (ndarray): u vector with shape (time, z, y, x).
#     v (ndarray): v vector with shape (time, z, y, x).
#     w (ndarray): w vector with shape (time, z, y, x).

#     Returns:
#     tuple: Reordered u, v, w vectors with shape (y, x, z, t).
#     """

#     # Transpose the arrays to reorder them from (time, z, y, x) to (y, x, z, time)
#     u_reordered = np.transpose(u, (2, 3, 1, 0))  # Change axes to (y, x, z, t)
#     v_reordered = np.transpose(v, (2, 3, 1, 0))
#     w_reordered = np.transpose(w, (2, 3, 1, 0))
    
#     # u_reordered = np.transpose(u, (3, 2, 1, 0))  # Change axes to (x, y, z, t)
#     # v_reordered = np.transpose(v, (3, 2, 1, 0))
#     # w_reordered = np.transpose(w, (3, 2, 1, 0))
    
#     return u_reordered, v_reordered, w_reordered

# u, v, w = reorder_velocity(u_old, v_old, w_old)

# # print("Reordered u shape:", u.shape)
# # print("Reordered v shape:", v.shape)
# # print("Reordered w shape:", w.shape) #(100, 100,100,22)



# xmin = x.min()+10
# xmax = xu.max()-10
# ymin = y.min()+10
# ymax = yv.max()-10
# # zmin = 500
# # zmax = 500
# zmin = zu_3d.min()+10
# zmax = zu_3d.max()-10


# X,Y,Z = np.meshgrid(x,y,zu_3d)
# dx_data = X[0,1,0]-X[0,0,0] 
# dy_data = Y[1,0,0]-Y[0,0,0] 
# dz_data = Z[0,0,1]-Z[0,0,0] 

# delta = [dx_data, dy_data, dx_data]
# aux_grid = [delta[0], delta[1], delta[2]]
# #aux_grid = [200, 200, 200]


# periodic_x = False
# periodic_y = False
# periodic_z = False
# periodic = [periodic_x, periodic_y, periodic_z]
# bool_unsteady = True

# Ny = 50
# Nx = 50
# Nz = 50

# x_domain = np.linspace(xmin, xmax, Nx, endpoint = True) # array (Nx, )
# y_domain = np.linspace(ymin, ymax, Ny, endpoint = True) # array (Ny, )
# z_domain = np.linspace(zmin, zmax, Nz, endpoint = True) # array (Nz, )


# dx = x_domain[1]-x_domain[0] # float
# dy = y_domain[1]-y_domain[0] # float
# dz = z_domain[1]-z_domain[0] # float
# print(dx)

# X_domain, Y_domain, Z_domain = np.meshgrid(x_domain, y_domain, z_domain) 

# Ny = X_domain.shape[0] # int
# Nx = X_domain.shape[1] # int
# Nz = X_domain.shape[2] # int



# import numpy.ma as ma
# is_masked = ma.isMaskedArray(u)

# if is_masked:
#     u = u.filled(0)
# nan_indices = np.where(np.isnan(u))
# #print("NaN values u:", nan_indices)


# is_masked = ma.isMaskedArray(v)
# if is_masked:
#     v = v.filled(0)
# nan_indices = np.where(np.isnan(v))
# #print("NaN values v:", nan_indices)

# is_masked = ma.isMaskedArray(w)
# if is_masked:
#     w = w.filled(0)
# nan_indices = np.where(np.isnan(w))
# #print("NaN values w:", nan_indices)

# time_data = time.reshape(1,-1)

# Interpolant = interpolant_unsteady(X, Y, Z, u, v, w, time_data)

# Interpolant_u = Interpolant[0] # RectangularBivariateSpline-object
# Interpolant_v = Interpolant[1] # RectangularBivariateSpline-object
# Interpolant_w = Interpolant[2] # RectangularBivariateSpline-object



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
#time = time[np.newaxis, :]  # time shape (1,100)

# print("Reordered u shape:", u.shape)
# print("Reordered v shape:", v.shape)
# print("Reordered w shape:", w.shape) #(100, 100,100,22)



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
bool_unsteady = True

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


dx = x_domain[1]-x_domain[0] # float
dy = y_domain[1]-y_domain[0] # float
#dz = z_domain[1]-z_domain[0] # float
dz_x, dz_y = np.gradient(Z_domain, dx, dy)
dz = np.mean(np.sqrt(dz_x**2 + dz_y**2))

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
tN = time[10] # float

# Time step-size
dt = 0.1 # float

time = np.arange(t0, tN+dt, dt) # shape (Nt,)

# Length of time interval
lenT = abs(tN-t0)


def Fmap_test(x0, y0, z0):
    
    X0 = np.array([x0, y0, z0]) # array (3, Nx*Ny*Nz)
    
    Fmap = integration_dFdt(time, X0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)[0] # array (Nt, 3, Nx*Ny*Nz)
    
    return Fmap,X0
########### without batches!
# xy
# x0_xy = X_domain[:, :, -1].ravel()   # (Nx*Ny,)
# y0_xy = Y_domain[:, :, -1].ravel() 
# z0_xy = Z_domain[:, :, -1].ravel()  

x0_xy = X_domain.ravel()   # (Nx*Ny,)
y0_xy = Y_domain.ravel() 
z0_xy = Z_domain.ravel() 
#test,xx0 = Fmap_test(x0_xy, y0_xy, z0_xy)

def velocity_test(x0, y0, z0):
   
    X0 = np.array([x0, y0, z0])  # (3, Npoints)
    veloc = np.zeros((len(time), 3, X0.shape[1])) 
    
    
    for i in range(len(time)):
        
        veloc[i, :, :] = velocity(time[i], X0, X, Y, Z, Interpolant_u, Interpolant_v, Interpolant_w, periodic, bool_unsteady)
    
    return veloc, X0

#vv, X_initial = velocity_test(x0_xy, y0_xy, z0_xy)


def jacobian(vel, dx, dy, dz):
    Nt, _, Npoints = vel.shape  
    jacobians = np.zeros((Nt, Npoints, 3, 3))  

    for t in range(Nt):
        
        u = vel[t, 0, :]  
        v = vel[t, 1, :]  
        w = vel[t, 2, :] 
        
        
        dudx = np.gradient(u, dx)  # ∂u/∂x
        dudy = np.gradient(u, dy)  # ∂u/∂y
        dudz = np.gradient(u, dz)  # ∂u/∂z
        
        dvdx = np.gradient(v, dx)  # ∂v/∂x
        dvdy = np.gradient(v, dy)  # ∂v/∂y
        dvdz = np.gradient(v, dz)  # ∂v/∂z
        
        dwdx = np.gradient(w, dx)  # ∂w/∂x
        dwdy = np.gradient(w, dy)  # ∂w/∂y
        dwdz = np.gradient(w, dz)  # ∂w/∂z
        
        
        for i in range(Npoints):
            jacobians[t, i, 0, 0] = dudx[i]  # ∂u/∂x
            jacobians[t, i, 0, 1] = dudy[i]  # ∂u/∂y
            jacobians[t, i, 0, 2] = dudz[i]  # ∂u/∂z
            jacobians[t, i, 1, 0] = dvdx[i]  # ∂v/∂x
            jacobians[t, i, 1, 1] = dvdy[i]  # ∂v/∂y
            jacobians[t, i, 1, 2] = dvdz[i]  # ∂v/∂z
            jacobians[t, i, 2, 0] = dwdx[i]  # ∂w/∂x
            jacobians[t, i, 2, 1] = dwdy[i]  # ∂w/∂y
            jacobians[t, i, 2, 2] = dwdz[i]  # ∂w/∂z
    
    return jacobians


#J = jacobian(vv, dx, dy, dz)

#plt.plot(J[0,:,0,:], J[0,:,1,:])

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



# jac = []

# for t in range(vv.shape[0]):

#     u_t = vv[t, 0, :] 
#     v_t = vv[t, 1, :] 
#     w_t = vv[t, 2, :]
    
#     u_t = u_t.reshape(Nx, Ny, Nz)
#     v_t = v_t.reshape(Nx, Ny, Nz)
#     w_t = w_t.reshape(Nx, Ny, Nz)
    
  
#     dudx, dudy, dudz = np.gradient(u_t, dx, dy, dz, edge_order=2)
#     dvdx, dvdy, dvdz = np.gradient(v_t, dx, dy, dz, edge_order=2)
#     dwdx, dwdy, dwdz = np.gradient(w_t, dx, dy, dz, edge_order=2)
    
  
#     jac_t = np.stack((
#         np.stack((dudx, dudy, dudz), axis=-1),
#         np.stack((dvdx, dvdy, dvdz), axis=-1),
#         np.stack((dwdx, dwdy, dwdz), axis=-1)
#     ), axis=-2) 
    
#     jac.append(jac_t)


# jac = np.array(jac)

def strain_tensor(jacobians):
    
    S = 0.5 * (jacobians + np.transpose(jacobians, (0, 1, 2, 3, 5, 4)))  
    #S = 0.5 * (jacobians + np.transpose(jacobians, (0, 1, 3, 2)))  
    return S

S = strain_tensor(jacobians)
#S_vel = strain_tensor(J)

print(dx, dy, dz)  



from matplotlib.animation import FuncAnimation
%matplotlib qt

# S_xx = S_vel[:,:, 0, 0]  

# S_xx_over_time_reshaped = S_xx.reshape(len(time), Nx, Ny)  # Shape: (182, 50, 50)

# fig, ax = plt.subplots(figsize=(8, 6))
# cax = ax.imshow(S_xx_over_time_reshaped[0], cmap='viridis', vmin=np.min(S_xx), vmax=np.max(S_xx))
# fig.colorbar(cax, ax=ax, label="S_xx")
# ax.set_title("Strain Tensor Component S_xx over Time")
# ax.set_xlabel("X Position")
# ax.set_ylabel("Y Position")


# def update(frame):
#     cax.set_data(S_xx_over_time_reshaped[frame]) 
#     ax.set_title(f"Strain Tensor Component S_xx at Time Step {frame}")
#     return cax,


# anim = FuncAnimation(fig, update, frames=182, interval=100, blit=True)
# plt.show()



### Time development

# x_point, y_point = 50, 50 
# z_level = 50               


# S_xx_time_series = S[:, x_point, y_point, z_level, 1, 1]  
# plt.figure(figsize=(8, 6))
# plt.plot(range(S.shape[0]), S_xx_time_series)
# plt.xlabel("Time Index")
# plt.ylabel("S_xx")
# plt.grid()
# plt.show()



############# Time evolution of strain tensor
# z_index = 5
# S_xx = S[..., 0, 0]  
# S_xx_slice = S_xx[:, :, :, z_index]  # shape (time, y, x)


# x_domain_Sxx = np.linspace(x.min(), x.max(), S_xx_slice.shape[2])
# y_domain_Sxx = np.linspace(y.min(), y.max(), S_xx_slice.shape[1])


# fig, ax = plt.subplots(figsize=(8, 6))
# contour = ax.contourf(x_domain_Sxx, y_domain_Sxx, S_xx_slice[0], cmap='viridis', levels=100)
# colorbar = fig.colorbar(contour, ax=ax, label="S_xx")
# title = ax.text(0.5, 1.05, f"Strain S_xx at t=0", ha="center", va="center", transform=ax.transAxes)


# def update(frame):
#     ax.clear()  
#     contour = ax.contourf(x_domain_Sxx, y_domain_Sxx, S_xx_slice[frame], cmap='viridis', levels=100)
#     title.set_text(f"Strain S_xx at t={frame}")
#     return contour, title

# ani = FuncAnimation(fig, update, frames=S_xx_slice.shape[0], interval=200, blit=False)
# plt.show()

# t_index = 50


# x_domain_Sxx = np.linspace(x.min(), x.max(), S_xx_slice.shape[2])
# y_domain_Sxx = np.linspace(y.min(), y.max(), S_xx_slice.shape[1])


# fig, ax = plt.subplots(figsize=(8, 6))
# contour = ax.contourf(x_domain_Sxx, y_domain_Sxx, S_xx_slice[t_index], cmap='viridis', levels=100)
# colorbar = fig.colorbar(contour, ax=ax, label="S_xx")
# ax.set_title(f"Strain S_xx at t={t_index}")
# ax.set_xlabel("x")
# ax.set_ylabel("y")

# plt.show()



def eigenvalues_strain(S):
    Nt, Npoints, _, _ = S.shape
    eigenvalues = np.zeros((Nt, Npoints, 3))  
    
    for t in range(Nt):
        for i in range(Npoints):
            
            strain_matrix = S[t, i, :, :]
            eigvals = np.linalg.eigvals(strain_matrix)
            eigenvalues[t, i, :] = eigvals
    
    return eigenvalues


#eigenvals = eigenvalues_strain(S)
eigen = np.linalg.eigvals(S) #shape (54, 100, 100, 82, 3)



max_eigen_z = np.max(eigen, axis=-2)  # shape (54, 100, 100, 3)
max_eigen = np.max(max_eigen_z, axis=-1)  # shape: (54, 100, 100)


t_index = 20  

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




# max_eigen = np.max(eigen, axis=-1)

# t_index = 0
# z_index = 20
# max_eigen_slice = max_eigen[:,:,:, z_index]

# x_domain = np.linspace(x.min(), x.max(), max_eigen_slice.shape[2])
# y_domain = np.linspace(y.min(), y.max(), max_eigen_slice.shape[1])


# fig, ax = plt.subplots(figsize=(8, 6))
# contour = ax.contourf(x_domain, y_domain, max_eigen_slice[t_index], cmap='viridis', levels=100)
# colorbar = fig.colorbar(contour, ax=ax, label = 'Eigenvalues')

# plt.show()



