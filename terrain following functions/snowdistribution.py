#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 15:01:42 2025

@author: sarahvalent
"""


import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
##### ullstinden
### test from 45°
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/test from 45 grad/Ullstinden_Recenter_topo_20_crs_45_7_20m.shp'

#nc_file = '/Users/sarahvalent/Desktop/MASTER-THESIS/RESULT ANALYSIS/ullstinden/Ullstinden_Recenter_av_masked_N03_M01.001.nc'

#averaged
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/averaged_ ullstinden 20m/Ullstinden_Recenter_topo_20_crs_225_1_20m.shp'
#shapefile_path ='/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/averaged_ ullstinden 20m/7 ms/Ullstinden_Recenter_topo_20_crs_225_7_20m.shp'
# weatherstation
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/Ullstinden_Recenter_topo_20_crs_point_03-19-2025_1503_20m.shp'
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/multiple weatherstations/Ullstinden_Recenter_topo_20_crs_point_03-19-2025_1503_20m.shp'
shapefile_path =  '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/neue stationen/Ullstinden_Recenter_topo_20_crs_point_03-31-2025_1924_20m.shp'

## toymountain 
nc_file ='/Users/sarahvalent/Desktop/MASTER-THESIS/RESULT ANALYSIS/toymountain/PALM 20m/toy_mountain_smooth_inflow_20m_av_masked_M01.000.nc'
#averaged
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resoultiuon 20m DEM _averaged/toy_mountain_smooth_inflow_20m_topo_270_1_20m.shp'
#weather station
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM_ weatherstations/new with xy/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'
#one weather station
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM _ one weather station/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM _ one weather station/with multipe 270/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'


###
rho_air=1.34 # fluid density
rho_snow=140 # particle density
C_0=0.6
delta_0=5.83
L=1000 #m, 
d=100*10**(-6) #m,
U=1.2 #m/s, 
Re=10**8
alpha = np.sqrt(d/L)
beta = 3*rho_air / (2*rho_snow + rho_air)
T = L/U
g = 9.81 * (T**2/L)
##



class SNOWdistribution:
    def __init__(self, data_source, is_netcdf=False, Nx=200, Ny=200, U = 1.2, L = 1000, delta=0.01):
        self.is_netcdf = is_netcdf
        self.Nx, self.Ny = Nx, Ny
        self.U = U
        self.L = L
        self.delta = delta

        if is_netcdf:
            self.load_netcdf(data_source)
        else:
            self.load_shapefile(data_source)

        self.create_interpolants()
    
    def load_shapefile(self, shapefile_path):
        gdf = gpd.read_file(shapefile_path)
        gdf['x'] = gdf.geometry.x
        gdf['y'] = gdf.geometry.y

        self.x_unique = (np.sort(gdf['x'].unique())) / self.L
        self.y_unique = (np.sort(gdf['y'].unique()) ) / self.L

        speed_grid = np.zeros((len(self.y_unique), len(self.x_unique)))
        dir_grid = np.zeros((len(self.y_unique), len(self.x_unique)))

        for _, row in gdf.iterrows():
            x_idx = np.where(self.x_unique == row['x'] /self.L)[0][0]
            y_idx = np.where(self.y_unique == row['y'] /self.L)[0][0]
            speed_grid[y_idx, x_idx] = row['speed']
            dir_grid[y_idx, x_idx] = row['dir']

        self.u_grid =( -speed_grid * np.sin(np.radians(dir_grid))) / self.U
        self.v_grid =( -speed_grid * np.cos(np.radians(dir_grid))) / self.U


    def load_netcdf(self, nc_file):
            import netCDF4 as nc
            dataset = nc.Dataset(nc_file, 'r')

            u_data = dataset.variables['u'][-1, 1, :, :]
            v_data = dataset.variables['v'][-1, 1, :, :]
            grid_size = 20  # 20m resolution per grid cell
            x_coords = np.linspace(0, grid_size * (u_data.shape[1] - 1), u_data.shape[1])
            y_coords = np.linspace(0, grid_size * (u_data.shape[0] - 1), u_data.shape[0])


            self.u_grid = u_data / self.U
            self.v_grid = v_data / self.U
            self.x_unique = x_coords / self.L
            self.y_unique = y_coords / self.L

    def create_interpolants(self):
        self.u_interpolator = RegularGridInterpolator((self.y_unique, self.x_unique), self.u_grid)
        self.v_interpolator = RegularGridInterpolator((self.y_unique, self.x_unique), self.v_grid)
        self.x_domain = np.linspace(self.x_unique.min()+0.01, self.x_unique.max()-0.01, self.Nx)
        self.y_domain = np.linspace(self.y_unique.min()+0.01, self.y_unique.max()-0.01, self.Ny)
        self.X_domain, self.Y_domain = np.meshgrid(self.x_domain, self.y_domain)
        self.X0 = np.array([self.X_domain.ravel(), self.Y_domain.ravel()])

    def velocity(self, x):
        
        x = np.atleast_2d(x)  
        if x.shape[0] != 2:  
            x = x.T  
    
        if x.ndim != 2 or x.shape[0] != 2: 
            raise ValueError(f"Input x has incorrect shape {x.shape}, expected (2, Npoints)")
    
        x_swap = np.zeros((x.shape[1], 2))  
        x_swap[:, 0] = x[1, :]
        x_swap[:, 1] = x[0, :]
    
        u = self.u_interpolator(x_swap)
        v = self.v_interpolator(x_swap)
    
        return np.array([u, v])
    def compute_jacobian(self):
        
        num_points = self.X0.shape[1]
        jac_terrain_2d = np.zeros((2, 2, num_points))  # Shape (2,2,N)
    
        for i in range(num_points):
            jac_terrain_2d[:, :, i] = self._compute_jacobian_at_point(self.X0[:, i])

        return jac_terrain_2d  
    
    def _compute_jacobian_at_point(self, x):
        
        x0, y0 = x[0], x[1]
    

        xr, xl = np.array([x0 + self.delta, y0]), np.array([x0 - self.delta, y0])
        yu, yd = np.array([x0, y0 + self.delta]), np.array([x0, y0 - self.delta])
    
   
        vL, vR = self.velocity(xl), self.velocity(xr)
        vD, vU = self.velocity(yd), self.velocity(yu)
    
        dudx = (vR[0] - vL[0]) / (2 * self.delta)
        dudy = (vU[0] - vD[0]) / (2 * self.delta)
        dvdx = (vR[1] - vL[1]) / (2 * self.delta)
        dvdy = (vU[1] - vD[1]) / (2 * self.delta)
    
        jacobian = np.array([[dudx, dudy], [dvdx, dvdy]])
    
        return np.squeeze(jacobian)  
    def compute_advective_term(self):
            
            num_points = self.X0.shape[1]
            advective_term = np.zeros((2, num_points))  # Shape (2, N)
            
            jac_terrain_2d = self.compute_jacobian()  # Shape (2, 2, N)
            
            for i in range(num_points):
                v = np.squeeze(self.velocity(self.X0[:, i]) ) # Shape (2,)
                J = jac_terrain_2d[:, :, i]       # Shape (2, 2)
                advective_term[:, i] = J @ v      # (v ⋅ ∇)v
            
            return advective_term  # Shape (2, N)
    def perturbation(self):
        num_points = self.X0.shape[1]
        x_dot = np.zeros((2,num_points))
        adv_term = self.compute_advective_term()
        for i in range(num_points):
            adv = adv_term[:, i]
            A = (2 * Re / (beta * C_0 * delta_0**2)) * (1 - beta)
            B = (2 * np.sqrt(Re)) / delta_0
            
            u = np.squeeze(self.velocity(self.X0[:, i]) )
            u_4 = A* (g - adv)
            u_7 = - B * np.sqrt(np.abs(u_4))*u_4
            
            x_dot[:,i] = u + alpha**4*u_4 +alpha**7*u_7
        return x_dot

    def LAD(self):
        
        x_dot = self.perturbation()  
        num_points = x_dot.shape[1]
        LAD_vals = np.zeros((num_points))
        x_dot_u_component = x_dot[0].reshape((self.Ny,self. Nx))
        x_dot_v_component = x_dot[1].reshape((self.Ny,self. Nx))
        
        u_dot_interpolator = RegularGridInterpolator((self.y_domain, self.x_domain), x_dot_u_component, bounds_error=True)
        v_dot_interpolator = RegularGridInterpolator((self.y_domain, self.x_domain), x_dot_v_component, bounds_error=True)
      
        LAD_vals = np.full((self.Ny, self.Nx), np.nan)  # LAD grid

        for j in range(self.Ny):
            for i in range(self.Nx):
                x0 = self.x_domain[i]
                y0 = self.y_domain[j]
    
                try:
                    
                    xr = [y0, x0 + self.delta]
                    xl = [y0, x0 - self.delta]
                    yu = [y0 + self.delta, x0]
                    yd = [y0 - self.delta, x0]
    
                    
                    dudx = (u_dot_interpolator(xr) - u_dot_interpolator(xl)) / (2 * self.delta)
                    dudy = (u_dot_interpolator(yu) - u_dot_interpolator(yd)) / (2 * self.delta)
                    dvdx = (v_dot_interpolator(xr) - v_dot_interpolator(xl)) / (2 * self.delta)
                    dvdy = (v_dot_interpolator(yu) - v_dot_interpolator(yd)) / (2 * self.delta)
    
                    
                    J = np.array([[dudx, dudy], [dvdx, dvdy]])
    
                    
                    LAD_vals[j, i] = np.trace(J) 
    
                except ValueError:
                    
                    LAD_vals[j, i] = np.nan
    
        return LAD_vals
                
            
        
    
                
            
        
        
    
    
snow_palm = SNOWdistribution(nc_file, is_netcdf=True)
adv_term_palm = snow_palm.LAD()
lad_m_s = adv_term_palm / T

X, Y = np.meshgrid(snow_palm.x_domain*L, snow_palm.y_domain*L)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, lad_m_s.reshape(X.shape), cmap='viridis', levels = 250)
plt.xlabel('X (m)', fontsize=16)
plt.ylabel('Y (m)', fontsize=16)
plt.title('LAD PALM', fontsize=16)
cbar = plt.colorbar(contour)
cbar.set_label('LAD (1/s)', fontsize=16)
cbar.ax.tick_params(labelsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


snow_wn = SNOWdistribution(shapefile_path, is_netcdf=False)
adv_term_wn = snow_wn.compute_advective_term()
adv_term_palm = snow_wn.LAD()
lad_m_s = adv_term_palm / T

X, Y = np.meshgrid(snow_wn.x_domain*L, snow_wn.y_domain*L)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, lad_m_s.reshape(X.shape), levels=250, cmap='viridis')
plt.xlabel('X (m)', fontsize=16)
plt.ylabel('Y (m)', fontsize=16)
plt.title('LAD Windninja', fontsize=16)
cbar = plt.colorbar(contour)
cbar.set_label('LAD (1/s)', fontsize=16)
cbar.ax.tick_params(labelsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.show()

