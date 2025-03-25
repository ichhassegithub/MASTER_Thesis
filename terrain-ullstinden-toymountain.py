#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 08:15:44 2025

@author: sarahvalent
"""

#### Terrain plots

import rasterio
import numpy as np
import matplotlib.pyplot as plt

### Toymountain

#parameters
c1 = np.sqrt(0.125/ 2)
c2 = np.sqrt(1.25/ 2)

#domain
x = np.linspace(-2.500, 2.500, 100)
y = np.linspace(-2.500, 2.500, 100)
X, Y = np.meshgrid(x, y)

#equations
ridges_only = 0.5* np.exp(-(X**2) / (2 * c1**2)) + 0.5* np.exp(-(Y**2) / (2 * c1**2))
mountain_topo = ridges_only**0.3 * np.exp(-(X**2) / (2 * c2**2) - (Y**2) / (2 * c2**2))
Z = mountain_topo

#save file
# output_file = 'mountain_heights_original.txt'
# np.savetxt(output_file, Z, fmt='%.4f')  

#slope angle
dZdx, dZdy = np.gradient(Z, x, y)
gradient_magnitude = np.sqrt(dZdx**2 + dZdy**2)

slope_angle_degrees = np.arctan(gradient_magnitude) * (180 / np.pi)
max_slope_angle = np.max(slope_angle_degrees)
print(f"Maximum slope angle: {max_slope_angle:.4f} degrees")
print(np.max(Z))

#plot figure
fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
cbar1.set_label('Elevation (km)',fontsize=14)


ax1.set_xlabel('X (km)',fontsize=14)
ax1.set_ylabel('Y (km)',fontsize=14)
ax1.set_zlabel('Z (km)',fontsize=14)
ax1.set_title('Mountain height',fontsize=14)
#ax1.set_box_aspect([2.5, 2.5, 1])

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(slope_angle_degrees / np.max(slope_angle_degrees)), edgecolor='none')


m = plt.cm.ScalarMappable(cmap='viridis')
m.set_array(slope_angle_degrees)
cbar2 = fig.colorbar(m, ax=ax2, shrink=0.5, aspect=5)
cbar2.set_label('Slope Angle (Degrees)',fontsize=14)


ax2.set_xlabel('X (km)',fontsize=14)
ax2.set_ylabel('Y (km)',fontsize=14)
ax2.set_zlabel('Z (km)',fontsize=14)
ax2.set_title('Slope angle',fontsize=14)
#ax2.set_box_aspect([2.5, 2.5, 1])

plt.show()

### Ullstinden

file_tif ='/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/Ullstinden_Recenter_topo_20_crs.tif'
#open and read file
with rasterio.open(file_tif) as src_tif:
    nodata_value = src_tif.nodata  
    tif_data = src_tif.read(1)  
    transform = src_tif.transform  

#look for nan data
tif_data = np.where(tif_data == nodata_value, np.nan, tif_data)

#get meters
xmin, ymax = transform * (0, 0)  
xmax, ymin = transform * (tif_data.shape[1], tif_data.shape[0])  

#plot figure 
plt.figure(figsize=(8, 6))
plt.imshow(tif_data, cmap='terrain', extent=[xmin, xmax, ymin, ymax], origin="upper")  


cbar = plt.colorbar(label="Elevation (m)")  
cbar.ax.tick_params(labelsize=12)  
cbar.set_label("Elevation (m)", fontsize=14) 


plt.title("Elevation Map - Ullstinden", fontsize=14)
plt.xlabel("X (m)", fontsize=14)
plt.ylabel("Y (m)", fontsize=14)

plt.show()