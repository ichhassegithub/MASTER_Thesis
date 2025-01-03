#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:48:54 2024

@author: sarahvalent
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.interpolate import RegularGridInterpolator


# file_u = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_10/ref-DEM-proj_0_U.tif'
# file_v = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_10/ref-DEM-proj_0_V.tif'
# file_speed_up = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_10/ref-DEM-proj_0_spd_up_tile.tif'
# topography = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_10/ref-DEM-proj.tif'

# file_u = '//Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_50/ref-DEM-proj_315_U.tif'
# file_v = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_50/ref-DEM-proj_315_V.tif'
# file_speed_up = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_50/ref-DEM-proj_315_spd_up_tile.tif'
# topography = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_50/ref-DEM-proj.tif'


# file_u = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_100/ref-DEM-proj_90_U.tif'
# file_v = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_100/ref-DEM-proj_90_V.tif'
# file_speed_up = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_100/ref-DEM-proj_90_spd_up_tile.tif'
# topography = '/Users/sarahvalent/Desktop/MASTER-THESIS/WINDMAPPER/windmapper_2/wm_100/ref-DEM-proj.tif'



with rasterio.open(file_u) as src_u, rasterio.open(file_v) as src_v, rasterio.open(file_speed_up) as scr_speed, rasterio.open(topography) as scr_topo:
    u_data = src_u.read(1)
    v_data = src_v.read(1)
    speed_up_file = scr_speed.read(1)
    topo_file = scr_topo.read(1)
    transform = src_u.transform

u_data = np.array(np.abs(u_data))
v_data = np.array(np.abs(v_data))
speed_up_file = np.array(np.abs(speed_up_file))
topo_file = np.array(topo_file)


u_data_reformed = np.where(u_data > 100, np.nan, u_data) 
v_data_reformed = np.where(v_data > 100, np.nan, v_data)
speed_up_file_reformed = np.where(speed_up_file > 100, np.nan, speed_up_file)
speed = np.sqrt(u_data_reformed**2 + v_data_reformed**2)

#speed
x_coords = np.arange(u_data_reformed.shape[1])  
y_coords = np.arange(u_data_reformed.shape[0])  
X, Y = np.meshgrid(x_coords, y_coords)

u_interpolant = RegularGridInterpolator((y_coords,x_coords),u_data_reformed)
v_interpolant = RegularGridInterpolator((y_coords,x_coords),v_data_reformed)

#topo
x_topo = np.linspace(x_coords.min(), x_coords.max(), topo_file.shape[1])  
y_topo = np.linspace(y_coords.min(), y_coords.max(), topo_file.shape[0])  
X_topo, Y_topo = np.meshgrid(x_topo, y_topo)

topo_interpolator = RegularGridInterpolator((y_topo, x_topo), topo_file)  
coords = np.c_[Y_topo.ravel(), X_topo.ravel()]  # Shape: (N, 2)
Z_flat = topo_interpolator(coords)  # Shape: (N,)
Z_topo = Z_flat.reshape(X_topo.shape)  # Reshape to (500, 500)

offset = 20
Z_topo += offset

Nx = 500
Ny = 500

x_domain = np.linspace(x_coords.min()+10, x_coords.max()-10, Nx)
y_domain = np.linspace(y_coords.min()+10, y_coords.max()-10, Ny)

X_domain, Y_domain = np.meshgrid(x_domain, y_domain)

x0 = X_domain.ravel()
y0 = Y_domain.ravel()
X0 = np.array([x0,y0]) #shape (2,Npoints)

#### surface plot

# plt.figure(figsize=(12, 8))

# plt.xlabel("X (m)")
# plt.ylabel("Y (m)")
# plt.title(' u wm_10/ref-DEM-proj_0_spd_up_tile')
# surface = plt.pcolormesh(X_topo, Y_topo, topo_file, shading='auto', cmap='viridis')
# cbar = plt.colorbar(surface)
# cbar.set_label("(m/s)")

# plt.show()



def vel(x, u_interpolant, v_interpolant):
    
    x_swap = np.zeros((x.shape[1], 2)) # shape (Npoints, 2)
    x_swap[:,0] = x[1,:]
    x_swap[:,1] = x[0,:]

    u = u_interpolant(x_swap)
    v = v_interpolant(x_swap)
    
    veloc = np.array([u, v])
    
    return veloc


def strain_tensor(jacobians):
    
    strain = 0.5 * (jacobians + np.transpose(jacobians, axes = (1, 0, 2)))
    return strain



def eigenvalues_strain(strain_tensor):
    
    shape, _, num_points = strain_tensor.shape
    eigenvals = np.zeros((shape, num_points))

    
    for p in range(num_points):
            eigenvals[ :, p] = np.linalg.eigvalsh(strain_tensor[:, :, p])
    
    return eigenvals


########## yellow surface ###


def Jacobian_float(x, u_interpolant, v_interpolant,delta):
    
    x0, y0 = x[0], x[1]

    xr = np.array([x0 + delta, y0, topo_interpolator((y0, x0 + delta))+offset])
    xl = np.array([x0 - delta, y0, topo_interpolator((y0, x0 - delta))+offset])
    
    yu = np.array([x0, y0 + delta, topo_interpolator((y0 + delta, x0))+offset])
    yd = np.array([x0, y0 - delta, topo_interpolator((y0 - delta, x0))+offset])
    
   
    vL = vel(xl, u_interpolant, v_interpolant)
    vR = vel(xr, u_interpolant, v_interpolant)
    vD = vel(yd, u_interpolant, v_interpolant)
    vU = vel(yu, u_interpolant, v_interpolant)
 
   
    dudx = (vR[0] - vL[0]) / (2 * delta)
    dudy = (vU[0] - vD[0]) / (2 * delta)
 

    dvdx = (vR[1] - vL[1]) / (2 * delta)
    dvdy = (vU[1] - vD[1]) / (2 * delta)
   

   
    jac2 = np.array([
        [dudx, dudy],
        [dvdx, dvdy]
        
    ])
  

    return jac2
def compute_jacobian(X0, delta):
 

    num_points = X0.shape[1]
    jac_terrain_2d = np.zeros((2, 2, num_points)) 
    jac_terrain_2d[:,:,:] =  Jacobian_float(X0, u_interpolant, v_interpolant, delta)
    
    return jac_terrain_2d

dx_data = X[0,1]-X[0,0]  # 1
delta = 0.1*dx_data      #0.1


JAC_windmapper = compute_jacobian(X0, delta) 
S_windmapper = strain_tensor(JAC_windmapper)
e_windmapper = eigenvalues_strain(S_windmapper)
e_windmapper_max = np.max(e_windmapper, axis = 0)
max_eigen_slice = e_windmapper_max.reshape(Nx, Nx)  # (grid_size, grid_size)


plt.figure(figsize=(8, 6))
contour = plt.contourf(x_domain, y_domain, max_eigen_slice, levels=500, cmap='viridis')
plt.colorbar(contour, label="Max Eigenvalue")

plt.title(f"Max Eigenvalue windmapper ")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()





