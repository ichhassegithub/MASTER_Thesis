import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc

class WindAnalysis:
    def __init__(self, data_source, is_netcdf=False, Nx=500, Ny=500):
        self.is_netcdf = is_netcdf
        self.Nx, self.Ny = Nx, Ny

        if is_netcdf:
            self.load_netcdf(data_source)
        else:
            self.load_shapefile(data_source)

        self.create_interpolants()
    
    def load_shapefile(self, shapefile_path):
        gdf = gpd.read_file(shapefile_path)
        gdf['x'] = gdf.geometry.x
        gdf['y'] = gdf.geometry.y

        self.x_unique = np.sort(gdf['x'].unique())
        self.y_unique = np.sort(gdf['y'].unique())

        speed_grid = np.zeros((len(self.y_unique), len(self.x_unique)))
        dir_grid = np.zeros((len(self.y_unique), len(self.x_unique)))

        for _, row in gdf.iterrows():
            x_idx = np.where(self.x_unique == row['x'])[0][0]
            y_idx = np.where(self.y_unique == row['y'])[0][0]
            speed_grid[y_idx, x_idx] = row['speed']
            dir_grid[y_idx, x_idx] = row['dir']

        self.u_grid = -speed_grid * np.sin(np.radians(dir_grid))
        self.v_grid = -speed_grid * np.cos(np.radians(dir_grid))

    # def load_netcdf(self, nc_file):
    #     dataset = nc.Dataset(nc_file, 'r')
    #     self.u_grid = dataset.variables['u'][-1, 1, :, :]
    #     self.v_grid = dataset.variables['v'][-1, 1, :, :]
    #     self.x_unique = np.arange(self.u_grid.shape[1])
    #     self.y_unique = np.arange(self.u_grid.shape[0])
    def load_netcdf(self, nc_file):
            import netCDF4 as nc
            dataset = nc.Dataset(nc_file, 'r')

            u_data = dataset.variables['u'][-1, 1, :, :]
            v_data = dataset.variables['v'][-1, 1, :, :]
            grid_size = 20  # 20m resolution per grid cell
            x_coords = np.linspace(0, grid_size * (u_data.shape[1] - 1), u_data.shape[1])
            y_coords = np.linspace(0, grid_size * (u_data.shape[0] - 1), u_data.shape[0])


            self.u_grid = u_data
            self.v_grid = v_data
            self.x_unique = x_coords
            self.y_unique = y_coords

    def create_interpolants(self):
        self.u_interpolator = RegularGridInterpolator((self.y_unique, self.x_unique), self.u_grid)
        self.v_interpolator = RegularGridInterpolator((self.y_unique, self.x_unique), self.v_grid)
        self.x_domain = np.linspace(self.x_unique.min()+10, self.x_unique.max()-10, self.Nx)
        self.y_domain = np.linspace(self.y_unique.min()+10, self.y_unique.max()-10, self.Ny)
        self.X_domain, self.Y_domain = np.meshgrid(self.x_domain, self.y_domain)
        self.X0 = np.array([self.X_domain.ravel(), self.Y_domain.ravel()])

    def velocity(self, x):
        """Compute interpolated velocity components at given points."""
        x = np.atleast_2d(x)  # Ensure x is at least 2D
        if x.shape[0] != 2:  
            x = x.T  # Transpose if needed
    
        if x.ndim != 2 or x.shape[0] != 2:  # Extra safety check
            raise ValueError(f"Input x has incorrect shape {x.shape}, expected (2, Npoints)")
    
        x_swap = np.zeros((x.shape[1], 2))  
        x_swap[:, 0] = x[1, :]
        x_swap[:, 1] = x[0, :]
    
        u = self.u_interpolator(x_swap)
        v = self.v_interpolator(x_swap)
    
        return np.array([u, v])
    def compute_jacobian(self, delta=0.001):
        """Compute Jacobian matrix for velocity field."""
        num_points = self.X0.shape[1]
        jac_terrain_2d = np.zeros((2, 2, num_points))  # Shape (2,2,N)
    
        for i in range(num_points):
            jac_terrain_2d[:, :, i] = self._compute_jacobian_at_point(self.X0[:, i], delta)

        return jac_terrain_2d  
    def _compute_jacobian_at_point(self, x, delta):
        """Compute Jacobian at a single point using finite differences."""
        x0, y0 = x[0], x[1]
    

        xr, xl = np.array([x0 + delta, y0]), np.array([x0 - delta, y0])
        yu, yd = np.array([x0, y0 + delta]), np.array([x0, y0 - delta])
    
   
        vL, vR = self.velocity(xl), self.velocity(xr)
        vD, vU = self.velocity(yd), self.velocity(yu)
    
        dudx = (vR[0] - vL[0]) / (2 * delta)
        dudy = (vU[0] - vD[0]) / (2 * delta)
        dvdx = (vR[1] - vL[1]) / (2 * delta)
        dvdy = (vU[1] - vD[1]) / (2 * delta)
    
        jacobian = np.array([[dudx, dudy], [dvdx, dvdy]])
    
        return np.squeeze(jacobian)  

    def compute_strain_tensor(self):
        jacobians = self.compute_jacobian()
        return 0.5 * (jacobians + np.transpose(jacobians, axes=(1, 0, 2)))
    
    def compute_eigenvalues(self):
       
        strain = self.compute_strain_tensor()
        num_points = strain.shape[2]
        eigenvalues = np.zeros((2, num_points))

        for i in range(num_points):
            eigenvalues[:, i] = np.linalg.eigvalsh(strain[:, :, i])

        return eigenvalues

    # def plot_eigenvalues(self, index=0):
        
    #     eigenvalues = self.compute_eigenvalues()
    #     eigenvalue_field = eigenvalues[index].reshape(self.Nx, self.Ny)

    #     plt.figure(figsize=(10, 6))
    #     plt.contourf(self.x_domain, self.y_domain, eigenvalue_field, levels=200, cmap='coolwarm')
    #     plt.colorbar(label=f'Eigenvalue {index}')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title(f'Strain Tensor Eigenvalue {index}')
    #     plt.show()

    # def plot_shear_component(self):
    #     strain = self.compute_strain_tensor()
    #     shear_component = strain[0, 1, :].reshape(self.Nx, self.Ny)
    #     plt.figure(figsize=(10, 6))
    #     plt.contourf(self.x_domain, self.y_domain, shear_component, levels=200, cmap='viridis')
    #     plt.colorbar(label='Shear Component S_xy')
    #     plt.title('Shear Component of Strain Tensor')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.show()

    # @classmethod
    # def plot_strain_tensor_difference(cls, dataset_1, dataset_2):
    #     strain_1 = dataset_1.compute_strain_tensor()
    #     strain_2 = dataset_2.compute_strain_tensor()
    #     strain_diff = strain_1 - strain_2
    #     diff_component = strain_diff[0, 1, :].reshape(dataset_1.Nx, dataset_1.Ny)
    #     plt.figure(figsize=(10, 6))
    #     plt.contourf(dataset_1.x_domain, dataset_1.y_domain, diff_component, levels=100, cmap='RdBu_r')
    #     plt.colorbar(label='Δ S_xy')
    #     plt.title('Difference in Shear Strain Component (S_xy)')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.show()


##### ullstinden
nc_file = '/Users/sarahvalent/Desktop/MASTER-THESIS/RESULT ANALYSIS/ullstinden/Ullstinden_Recenter_av_masked_N03_M01.001.nc'

#averaged
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/averaged_ ullstinden 20m/Ullstinden_Recenter_topo_20_crs_225_1_20m.shp'
#shapefile_path ='/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/averaged_ ullstinden 20m/7 ms/Ullstinden_Recenter_topo_20_crs_225_7_20m.shp'
# weatherstation
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/Ullstinden_Recenter_topo_20_crs_point_03-19-2025_1503_20m.shp'
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/multiple weatherstations/Ullstinden_Recenter_topo_20_crs_point_03-19-2025_1503_20m.shp'
shapefile_path =  '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/neue stationen/Ullstinden_Recenter_topo_20_crs_point_03-31-2025_1924_20m.shp'

## toymountain 
#nc_file ='/Users/sarahvalent/Desktop/MASTER-THESIS/RESULT ANALYSIS/toymountain/PALM 20m/toy_mountain_smooth_inflow_20m_av_masked_M01.000.nc'
#averaged
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resoultiuon 20m DEM _averaged/toy_mountain_smooth_inflow_20m_topo_270_1_20m.shp'
#weather station
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM_ weatherstations/new with xy/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'
#one weather station
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM _ one weather station/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM _ one weather station/with multipe 270/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'


shapefile_analysis = WindAnalysis(shapefile_path, is_netcdf=False)
netcdf_analysis = WindAnalysis(nc_file, is_netcdf=True)

def normalization(datafile):
    m =  np.mean(datafile)
    s = np.std(datafile)
    return (datafile - m)/s

eigenvalues_shapefile = shapefile_analysis.compute_eigenvalues()
eigenvalues_ncfile = netcdf_analysis.compute_eigenvalues()


norm_eigen_shapefile = normalization(eigenvalues_shapefile)
norm_eigen_ncfile = normalization(eigenvalues_ncfile)


norm_eigen_shapefile_reshaped = norm_eigen_shapefile[1].reshape(shapefile_analysis.Nx, shapefile_analysis.Ny)
norm_eigen_ncfile_reshaped = norm_eigen_ncfile[1].reshape(netcdf_analysis.Nx, netcdf_analysis.Ny)

vmax = max(np.abs(norm_eigen_shapefile).max(), np.abs(norm_eigen_ncfile).max())

plt.figure(figsize=(8, 6))
plt.contourf(shapefile_analysis.x_domain, shapefile_analysis.y_domain, norm_eigen_shapefile_reshaped, levels=200, cmap='seismic', vmax=vmax)
cbar = plt.colorbar(label='ROS largest Eigenvalue ')
cbar.ax.tick_params(labelsize=16)
plt.title('Eigenvalue Field (Windninja)', fontsize=16)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


plt.figure(figsize=(8, 6))
plt.contourf(netcdf_analysis.x_domain, netcdf_analysis.y_domain, norm_eigen_ncfile_reshaped, levels=200, cmap='seismic', vmax=vmax)
cbar = plt.colorbar(label='ROS largest Eigenvalue ')
cbar.ax.tick_params(labelsize=16)
plt.title('Normalized Eigenvalue Field (Palm)', fontsize=16)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()