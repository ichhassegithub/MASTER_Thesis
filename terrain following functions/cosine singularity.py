#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 16:53:25 2025

@author: sarahvalent
"""
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import svd
from numba import njit, prange
import matplotlib.colors as mcolors
import netCDF4 as nc
%matplotlib qt
##### ullstinden
#nc_file = '/Users/sarahvalent/Desktop/MASTER-THESIS/RESULT ANALYSIS/ullstinden/Ullstinden_Recenter_av_masked_N03_M01.001.nc'

#averaged
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/averaged_ ullstinden 20m/Ullstinden_Recenter_topo_20_crs_225_1_20m.shp'
#shapefile_path ='/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/averaged_ ullstinden 20m/7 ms/Ullstinden_Recenter_topo_20_crs_225_7_20m.shp'
# weatherstation
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/Ullstinden_Recenter_topo_20_crs_point_03-19-2025_1503_20m.shp'
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/multiple weatherstations/Ullstinden_Recenter_topo_20_crs_point_03-19-2025_1503_20m.shp'
#shapefile_path =  '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/neue stationen/Ullstinden_Recenter_topo_20_crs_point_03-31-2025_1924_20m.shp'

## toymountain 
nc_file ='/Users/sarahvalent/Desktop/MASTER-THESIS/RESULT ANALYSIS/toymountain/PALM 20m/toy_mountain_smooth_inflow_20m_av_masked_M01.000.nc'
#averaged
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resoultiuon 20m DEM _averaged/toy_mountain_smooth_inflow_20m_topo_270_1_20m.shp'
#weather station
shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM_ weatherstations/new with xy/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'
#one weather station
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM _ one weather station/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM _ one weather station/with multipe 270/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'


gdf = gpd.read_file(shapefile_path)
#print(gdf.head())

# gdf.plot(column='speed', cmap='viridis', legend=True)
# plt.title("Wind Speed")
# plt.show()

def grid_velocities(gdf):
   
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y

    x_unique = np.sort(gdf['x'].unique())  
    y_unique = np.sort(gdf['y'].unique())

    speed_grid = np.full((len(y_unique), len(x_unique)), 0.0)
    dir_grid = np.full((len(y_unique), len(x_unique)), 0.0)

    for _, row in gdf.iterrows():
        x_idx = np.where(x_unique == row['x'])[0][0]  
        y_idx = np.where(y_unique == row['y'])[0][0]  
        speed_grid[y_idx, x_idx] = row['speed']
        dir_grid[y_idx, x_idx] = row['dir']

    
    u_grid = -speed_grid * np.sin(np.radians(dir_grid))
    v_grid = -speed_grid * np.cos(np.radians(dir_grid))
    return u_grid, x_unique, y_unique, v_grid, speed_grid

u_grid, x_unique, y_unique,v_grid,speed = grid_velocities(gdf)



x_coords = x_unique
y_coords = y_unique


############################################################



data = nc.Dataset(nc_file, 'r')

print(data.variables.keys())


u = data.variables['u']
v = data.variables['v']


u_palm = u[-1,1,:,:] 
v_palm = v[-1,1,:,:]



###########################################################


def cosine_similarity (u_palm,v_palm,u_grid,v_grid):
    dot_prod = u_palm*u_grid + v_palm*v_grid
    palm = np.sqrt(u_palm**2+v_palm**2)
    wn = np.sqrt(u_grid**2 + v_grid**2)
    cs = dot_prod /(palm*wn)
    return np.nan_to_num(cs)

def plot(data, x_domain, y_domain):
    plt.figure(figsize=(8, 6))
    
    cf = plt.contourf(x_domain, y_domain, data, levels=250, cmap="seismic")
    
    cbar = plt.colorbar(cf)
    
    plt.title('Cosine singularity - with weatherstations', fontsize=16)
    plt.xlabel("X-axis", fontsize=16)
    plt.ylabel("Y-axis", fontsize=16)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add target points
    # target_points = [
    #     (2000, 8000), (8000, 8000), (2000, 2000), 
    #     (5200, 4100), (5840, 4450), (5400, 5300), 
    #     (6700, 7000), (6500, 6750), (5400, 6400)
    # ]
    # target_points = [(3000, 5000), (4900, 5500),(4900, 4500), (4300, 5000), (5400, 4500), (5400, 4000),(5400, 5000), (5400, 5500),(5400, 6000), (5400, 8000), (5400, 2000),(6400, 5900), (6400, 5000), (6400, 4100), (7300,5900), (7300, 5000), (7300, 4100)]
    # for x, y in target_points:
    #     plt.plot(x, y, 'w*', markersize=10)  # White star markers

    plt.show()

data = cosine_similarity (u_palm,v_palm,u_grid,v_grid)
plot (data, x_coords, y_coords)


# plt.figure(figsize=(8, 6))
# plt.contourf(x_coords, y_coords, u_palm, levels=250, cmap="seismic")
# plt.colorbar()
# plt.title("WindNinja U Component")
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.contourf(x_coords, y_coords, v_palm, levels=250, cmap="seismic")
# plt.colorbar()
# plt.title("WindNinja V Component")
# plt.show()










