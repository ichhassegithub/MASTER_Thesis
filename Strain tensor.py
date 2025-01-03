#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:40:06 2024

@author: sarahvalent
"""


import numpy as np
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from LAVD_setup_original_eth import *
from scipy.interpolate import griddata



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

#

# def create_topo_interpolator(x_topo, y_topo, topo_data, method='cubic'):
 
#     points = np.array([(y, x) for y in y_topo for x in x_topo])
#     values = topo_data.ravel()

#     def topo_interpolator(coords):
    
#         return griddata(points, values, coords, method=method)

#     return topo_interpolator


# topo_interpolator = create_topo_interpolator(x_topo, y_topo, topo_data)



X,Y,Z = np.meshgrid(x,y,zu_3d)
# dx_data = X[0,1,0]-X[0,0,0] 
# dy_data = Y[1,0,0]-Y[0,0,0] 
# dz_data = Z[0,0,1]-Z[0,0,0] 






Ny = 150
Nx = 150


x_domain = np.linspace(xmin, xmax, Nx, endpoint = True) # array (Nx, )
y_domain = np.linspace(ymin, ymax, Ny, endpoint = True) # array (Ny, )
#z_domain = np.linspace(zmin, zmax, Nz, endpoint = True) # array (Nz, )

X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
#Z_domain = topo_interpolator(np.c_[Y_domain.ravel(), X_domain.ravel()]).reshape(X_domain.shape)
coords = np.c_[Y_domain.ravel(), X_domain.ravel()]  # Shape: (N, 2)
Z_flat = topo_interpolator(coords)  # Shape: (N,)
Z_domain = Z_flat.reshape(X_domain.shape)  # Reshape to (150, 150)

offset = 400
Z_domain += offset






time_data = time.reshape(1,-1)

Interpolant = interpolant_unsteady(X, Y, Z, u, v, w, time_data)
#Interpolant = interpolant_unsteady(X, Y, Z, u, v, w, time_data, interpolation_method='cubic')

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
lenT = abs(tN-t0)


#### red surface ####

def vel(t, x, Interpolant_u, Interpolant_v, Interpolant_w):
    
    x_swap = np.zeros((x.shape[1], 4))
    x_swap[:,0] = x[1,:]
    x_swap[:,1] = x[0,:]
    x_swap[:,2] = x[2,:] 
    x_swap[:,3] = t
    

    u = Interpolant_u(x_swap)
    v = Interpolant_v(x_swap)
    w = Interpolant_w(x_swap)
    
    veloc = np.array([u, v, w])
    
    return veloc

def Jacobian_cartesian(t, x,Interpolant_u, Interpolant_v, Interpolant_w, delta):
           
    x = x.reshape(3,-1)
    
    X0, XL, XR, XU, XD, XF, XB = [], [], [], [], [], [], []
    
    for i in range(x.shape[1]):
        
        xr = x[0, i] + delta 
        xl = x[0, i] - delta
         
         
        yu = x[1, i] + delta
        yd = x[1, i] - delta
        
            
        zf = x[2, i] + delta
        zb = x[2, i] - delta
        
         
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
    
    vLend = vel(t, XL, Interpolant_u, Interpolant_v, Interpolant_w)    
    vRend = vel(t, XR, Interpolant_u, Interpolant_v, Interpolant_w)
    vDend = vel(t, XD, Interpolant_u, Interpolant_v, Interpolant_w)   
    vUend = vel(t, XU, Interpolant_u, Interpolant_v, Interpolant_w)
    vFend = vel(t, XF, Interpolant_u, Interpolant_v, Interpolant_w)   
    vBend = vel(t, XB, Interpolant_u, Interpolant_v, Interpolant_w)
    
    Jacobians = np.zeros((3, 3, X0.shape[1]))
    
    
    for i in range(X0.shape[1]):
        dudx = (vRend[0, i] - vLend[0, i]) / (2 * delta)
        dudy = (vUend[0, i] - vDend[0, i]) / (2 * delta)
        dudz = (vFend[0, i] - vBend[0, i]) / (2 * delta)
        
        dvdx = (vRend[1, i] - vLend[1, i]) / (2 * delta)
        dvdy = (vUend[1, i] - vDend[1, i]) / (2 * delta)
        dvdz = (vFend[1, i] - vBend[1, i]) / (2 * delta)
        
        dwdx = (vRend[2, i] - vLend[2, i]) / (2 * delta)
        dwdy = (vUend[2, i] - vDend[2, i]) / (2 * delta)
        dwdz = (vFend[2, i] - vBend[2, i]) / (2 * delta)
        
        
        Jacobians[:, :, i] = np.array([
            [dudx, dudy, dudz],
            [dvdx, dvdy, dvdz],
            [dwdx, dwdy, dwdz]])
    
    return Jacobians


def strain_tensor(jacobians):
    
    strain = 0.5 * (jacobians + np.transpose(jacobians, (0, 2, 1, 3)))
    return strain



def eigenvalues_strain(strain_tensor):
    
    time_steps, shape, _, num_points = strain_tensor.shape
    eigenvals = np.zeros((time_steps, shape, num_points))

    for t in range(time_steps):
        for p in range(num_points):
            eigenvals[t, :, p] = np.linalg.eigvalsh(strain_tensor[t, :, :, p])
    
    return eigenvals


########## yellow surface ###


def Jacobian_float(t, x, Interpolant_u, Interpolant_v, Interpolant_w, delta):
    
    x0, y0, z0 = x[0], x[1], x[2]

    xr = np.array([x0 + delta, y0, topo_interpolator((y0, x0 + delta))+offset])
    xl = np.array([x0 - delta, y0, topo_interpolator((y0, x0 - delta))+offset])
    
    yu = np.array([x0, y0 + delta, topo_interpolator((y0 + delta, x0))+offset])
    yd = np.array([x0, y0 - delta, topo_interpolator((y0 - delta, x0))+offset])
    
    ### ????
    zf = np.array([x0, y0, z0 + delta])
    zb = np.array([x0, y0, z0 - delta])

   
    vL = vel(t, xl, Interpolant_u, Interpolant_v, Interpolant_w)
    vR = vel(t, xr, Interpolant_u, Interpolant_v, Interpolant_w)
    vD = vel(t, yd, Interpolant_u, Interpolant_v, Interpolant_w)
    vU = vel(t, yu, Interpolant_u, Interpolant_v, Interpolant_w)
    vF = vel(t, zf, Interpolant_u, Interpolant_v, Interpolant_w)
    vB = vel(t, zb, Interpolant_u, Interpolant_v, Interpolant_w)

   
    dudx = (vR[0] - vL[0]) / (2 * delta)
    dudy = (vU[0] - vD[0]) / (2 * delta)
    dudz = (vF[0] - vB[0]) / (2 * delta)

    dvdx = (vR[1] - vL[1]) / (2 * delta)
    dvdy = (vU[1] - vD[1]) / (2 * delta)
    dvdz = (vF[1] - vB[1]) / (2 * delta)

    dwdx = (vR[2] - vL[2]) / (2 * delta)
    dwdy = (vU[2] - vD[2]) / (2 * delta)
    dwdz = (vF[2] - vB[2]) / (2 * delta)

   
    jac = np.array([
        [dudx, dudy, dudz],
        [dvdx, dvdy, dvdz],
        [dwdx, dwdy, dwdz]
    ])
    jac2 = np.array([
        [dudx, dudy],
        [dvdx, dvdy]
        
    ])
  

    return jac, jac2

x0_xy = X_domain.ravel()  
y0_xy = Y_domain.ravel() 
z0_xy = Z_domain.ravel()


def compute_jacobian_cartesian(x0_xy, y0_xy, z0_xy, delta):
 
    
    X0 = np.array([x0_xy, y0_xy, z0_xy])  
    #X02 = np.array([x0_xy, y0_xy]) 
    num_points = X0.shape[1]
    num_times = len(time)  
    
    jac_cartesian = np.zeros((num_times, 3, 3, num_points))  
   
    for i in range(num_times): 
        jac_cartesian[i, :, :, :] = Jacobian_cartesian(
            time[i], X0,  Interpolant_u, Interpolant_v, Interpolant_w, delta)
        

    return jac_cartesian


def compute_jacobian_all(x0_xy, y0_xy, z0_xy, delta):
 
    
    X0 = np.array([x0_xy, y0_xy, z0_xy])  
    X02 = np.array([x0_xy, y0_xy]) 
    num_points = X0.shape[1]
    num_times = len(time)  
    
    jac_cartesian = np.zeros((num_times, 3, 3, num_points))  
    jac_terrain_3d = np.zeros((num_times, 3, 3, num_points)) 
    jac_terrain_2d = np.zeros((num_times, 2, 2, num_points)) 

    for i in range(num_times): 
        jac_cartesian[i, :, :, :] = Jacobian_cartesian(
            time[i], X0,  Interpolant_u, Interpolant_v, Interpolant_w, delta)
        jac_terrain_3d[i, :, :, :], jac_terrain_2d[i, :, :, :] = Jacobian_float(
            time[i], X0,  Interpolant_u, Interpolant_v, Interpolant_w, delta)
       

    return jac_cartesian, jac_terrain_3d, jac_terrain_2d
dx_data = X[0,1,0]-X[0,0,0] 
delta = 0.1*dx_data

#JAC_cartesian = compute_jacobian_cartesian(x0_xy, y0_xy, z0_xy, delta)
JAC_cartesian, JAC_terrain_3d, JAC_terrain_2d = compute_jacobian_all(x0_xy, y0_xy, z0_xy, delta)

### cartesian
S_cartesian = strain_tensor(JAC_cartesian)
e_cartesian = eigenvalues_strain(S_cartesian)
#t_index = 20
max_eigen = np.max(e_cartesian, axis=1)  # shape: (182, Npoints)
grid_size = int(np.sqrt(max_eigen.shape[1]))  
#max_eigen_slice = max_eigen[t_index, :].reshape(grid_size, grid_size)

time_avg_eigen = np.mean(max_eigen, axis=0)  # time average (shape: Npoints)
max_eigen_slice = time_avg_eigen.reshape(grid_size, grid_size)  # (grid_size, grid_size)


# 2D
# S_terrain_following_2d = strain_tensor(JAC_terrain_2d)
# e_terrain_following_2d = eigenvalues_strain(S_terrain_following_2d)
# #t_index = 20
# max_eigen_terrain_2d = np.max(e_terrain_following_2d, axis=1)  # shape: (182, Npoints)
# grid_size = int(np.sqrt(max_eigen_terrain_2d.shape[1]))  
# #max_eigen_slice = max_eigen_terrain_2d[t_index, :].reshape(grid_size, grid_size)
# time_avg_eigen = np.mean(max_eigen_terrain_2d, axis=0)  # time average (shape: Npoints)
# max_eigen_slice = time_avg_eigen.reshape(grid_size, grid_size)  # (grid_size, grid_size)



## 3d
# S_terrain_following_3d = strain_tensor(JAC_terrain_3d)
# e_terrain_following_3d = eigenvalues_strain(S_terrain_following_3d)
# t_index = 20 
# max_eigen_terrain_3d = np.max(e_terrain_following_3d, axis=1)  # shape: (182, Npoints)
# grid_size = int(np.sqrt(max_eigen_terrain_3d.shape[1]))  
# max_eigen_slice = max_eigen_terrain_3d[t_index, :].reshape(grid_size, grid_size)

#### plot 


%matplotlib qt
x_domain = np.linspace(x0_xy.min(), x0_xy.max(), grid_size)
y_domain = np.linspace(y0_xy.min(), y0_xy.max(), grid_size)


plt.figure(figsize=(8, 6))
contour = plt.contourf(x_domain, y_domain, max_eigen_slice, levels=500, cmap='viridis')
plt.colorbar(contour, label="Max Eigenvalue")

plt.title(f"Max Eigenvalue cartesian ")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# grid_size = 500  # Increase this value for higher resolution
# x_domain = np.linspace(x0_xy.min(), x0_xy.max(), grid_size)
# y_domain = np.linspace(y0_xy.min(), y0_xy.max(), grid_size)
# X_grid, Y_grid = np.meshgrid(x_domain, y_domain)

# # Interpolate the max_eigen_slice onto the finer grid
# max_eigen_fine = griddata((x0_xy.flatten(), y0_xy.flatten()), max_eigen_slice.flatten(), (X_grid, Y_grid), method='cubic')

# # Plot the interpolated contour
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(X_grid, Y_grid, max_eigen_fine, levels=500, cmap='viridis')
# plt.colorbar(contour, label="Max Eigenvalue")

# plt.title(f"Max Eigenvalue Cartesian Linear at t={t_index}")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


#### cubic, soline interpolator and check offset
