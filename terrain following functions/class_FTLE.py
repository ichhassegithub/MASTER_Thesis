import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import svd
from numba import njit, prange
import matplotlib.colors as mcolors
import netCDF4 as nc

#%matplotlib qt
class FTLECalculator:
   
    
    def __init__(self, data_source, is_netcdf=False, Nx=500, Ny=500, delta=0.001, time=np.linspace(0, 0.1, 10)):
        self.is_netcdf = is_netcdf
        self.Nx = Nx
        self.Ny = Ny
        self.delta = delta
        self.time = time
        self.lenT = len(time)

        if is_netcdf:
            self.load_netcdf(data_source)
        else:
            self.load_shapefile(data_source)  # ✅ Now correctly passes data_source

        self.create_interpolants()

    def load_shapefile(self, shapefile_path):  # ✅ Now accepts shapefile_path
        gdf = gpd.read_file(shapefile_path)
        gdf['x'] = gdf.geometry.x
        gdf['y'] = gdf.geometry.y

        x_unique = np.sort(gdf['x'].unique()) 
        y_unique = np.sort(gdf['y'].unique())

        speed_grid = np.zeros((len(y_unique), len(x_unique)))
        dir_grid = np.zeros((len(y_unique), len(x_unique)))

        for _, row in gdf.iterrows():
            x_idx = np.where(x_unique == row['x'])[0][0]  
            y_idx = np.where(y_unique == row['y'])[0][0]  
            speed_grid[y_idx, x_idx] = row['speed']
            dir_grid[y_idx, x_idx] = row['dir']

        u_grid = -speed_grid * np.sin(np.radians(dir_grid))
        v_grid = -speed_grid * np.cos(np.radians(dir_grid))

        self.u_data = u_grid
        self.v_data = v_grid
        self.x_coords = x_unique
        self.y_coords = y_unique

    def load_netcdf(self, nc_file):
        import netCDF4 as nc
        dataset = nc.Dataset(nc_file, 'r')

        u_data = dataset.variables['u'][-1, 1, :, :]
        v_data = dataset.variables['v'][-1, 1, :, :]
        grid_size = 20  # 20m resolution per grid cell
        x_coords = np.linspace(0, grid_size * (u_data.shape[1] - 1), u_data.shape[1])
        y_coords = np.linspace(0, grid_size * (u_data.shape[0] - 1), u_data.shape[0])


        self.u_data = u_data
        self.v_data = v_data
        self.x_coords = x_coords
        self.y_coords = y_coords

    def create_interpolants(self):
        self.u_interpolant = RegularGridInterpolator((self.y_coords, self.x_coords), self.u_data, method='linear')
        self.v_interpolant = RegularGridInterpolator((self.y_coords, self.x_coords), self.v_data, method='linear')

        self.x_domain = np.linspace(self.x_coords.min() + 10, self.x_coords.max() - 10, self.Nx)
        self.y_domain = np.linspace(self.y_coords.min() + 10, self.y_coords.max() - 10, self.Ny)
        X_domain, Y_domain = np.meshgrid(self.x_domain, self.y_domain)

        self.x0 = X_domain.ravel()
        self.y0 = Y_domain.ravel()

    def compute_FTLE(self):
        X0 = np.array([self.x0, self.y0])  
        DF = self.gradient_flowmap(self.time, X0)
        ftle = [self.FTLE(DF[-1, :, :, i], self.lenT) for i in range(DF.shape[3])]
        return np.array(ftle).reshape(self.Ny, self.Nx)

    def integration_dFdt(self, time, x):
        x = x.reshape(2, -1)  
        Fmap = np.zeros((len(time), 2, x.shape[1]))  
        dFdt = np.zeros((len(time) - 1, 2, x.shape[1]))  
        dt = time[1] - time[0]  
        Fmap[0, :, :] = x

        for counter, t in enumerate(time[:-1]):
            Fmap[counter + 1, :, :], dFdt[counter, :, :] = self.RK4_step(t, Fmap[counter, :, :], dt)
        
        return Fmap, dFdt

    def RK4_step(self, t, x1, dt):
        x_prime = self.velocity(t, x1)
        k1 = dt * x_prime  
        x2 = x1 + 0.5 * k1  
        x_prime = self.velocity(t + 0.5 * dt, x2)
        k2 = dt * x_prime  
        x3 = x1 + 0.5 * k2  
        x_prime = self.velocity(t + 0.5 * dt, x3)
        k3 = dt * x_prime  
        x4 = x1 + k3  
        x_prime = self.velocity(t + dt, x4)
        k4 = dt * x_prime  

        y_prime_update = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)  
        return x1 + y_prime_update, y_prime_update / dt  

    def velocity(self, t, x):
        x_swap = np.zeros((x.shape[1], 2))
        x_swap[:, 0] = x[1, :]
        x_swap[:, 1] = x[0, :]

        u = self.u_interpolant(x_swap)
        v = self.v_interpolant(x_swap)
        return np.array([u, v])

    def gradient_flowmap(self, time, x):
        XL, XR, XU, XD = [], [], [], []

        for i in range(x.shape[1]):
            x0, y0 = x[0, i], x[1, i]
            XL.append([x0 - self.delta, y0])
            XR.append([x0 + self.delta, y0])
            XU.append([x0, y0 + self.delta])
            XD.append([x0, y0 - self.delta])

        XL = np.array(XL).T
        XR = np.array(XR).T
        XU = np.array(XU).T
        XD = np.array(XD).T

        XLend = self.integration_dFdt(time, XL)[0]  
        XRend = self.integration_dFdt(time, XR)[0]  
        XDend = self.integration_dFdt(time, XD)[0]  
        XUend = self.integration_dFdt(time, XU)[0]  

        return self.iterate_gradient(XRend, XLend, XUend, XDend)

    @staticmethod
    @njit(parallel=True)
    def iterate_gradient(XRend, XLend, XUend, XDend):
        gradFmap = np.zeros((XRend.shape[0], 2, 2, XRend.shape[2]))  

        for i in prange(XRend.shape[2]):      
            for j in prange(XRend.shape[0]):
                gradFmap[j, 0, 0, i] = (XRend[j, 0, i] - XLend[j, 0, i]) / (XRend[0, 0, i] - XLend[0, 0, i])
                gradFmap[j, 1, 0, i] = (XRend[j, 1, i] - XLend[j, 1, i]) / (XRend[0, 0, i] - XLend[0, 0, i])
                gradFmap[j, 0, 1, i] = (XUend[j, 0, i] - XDend[j, 0, i]) / (XUend[0, 1, i] - XDend[0, 1, i])
                gradFmap[j, 1, 1, i] = (XUend[j, 1, i] - XDend[j, 1, i]) / (XUend[0, 1, i] - XDend[0, 1, i])

        return gradFmap

    def FTLE(self, gradFmap, lenT):
        sigma_max = self.SVD(gradFmap)[1][0, 0]
        if sigma_max < 1:
            return 0
        return np.log(sigma_max) / lenT

    @staticmethod
    def SVD(gradFmap):
        P, s, QT = svd(gradFmap)
        return P, np.diag(s), QT.T

    def compare_FTLE(self, other):
        omega1 = self.compute_FTLE()
        omega2 = other.compute_FTLE()
        difference = omega1 - omega2
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
      
        cmap = plt.get_cmap("seismic")
        norm = mcolors.TwoSlopeNorm(vmin=difference.min(), vcenter=0, vmax=difference.max())
        
  
        contour = ax.contourf(self.x_domain, self.y_domain, difference, levels=250, cmap=cmap, norm=norm)
        colorbar = plt.colorbar(contour, ax=ax)
        colorbar.set_label("FTLE Difference")
        
        
        # rect = plt.Rectangle((4000, 4000), 1000, 2000, linewidth=1, edgecolor='r', facecolor='none', linestyle='--', label='average area')
        # ax.add_patch(rect)
        # rect2 = plt.Rectangle((6000, 4000), 1000, 2000, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
        # ax.add_patch(rect2)
        
        
        weather_stations = [
            (3000, 5000), (4900, 5500), (4900, 4500), (4300, 5000),
            (5400, 4500), (5400, 4000), (5400, 5000), (5400, 5500), 
            (5400, 6000), (5400, 8000), (5400, 2000), (6400, 5900),
            (6400, 5000), (6400, 4100), (7300, 5900), (7300, 5000),
            (7300, 4100)
        ]
        
        for x, y in weather_stations:
            ax.plot(x, y, 'k*')
        
        
        ax.set_title("FTLE Difference Map with Weather Stations")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

        plt.show()





##### ullstinden
nc_file = '/Users/sarahvalent/Desktop/MASTER-THESIS/RESULT ANALYSIS/ullstinden/Ullstinden_Recenter_av_masked_N03_M01.001.nc'

#averaged
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/averaged_ ullstinden 20m/Ullstinden_Recenter_topo_20_crs_225_1_20m.shp'
shapefile_path ='/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/averaged_ ullstinden 20m/7 ms/Ullstinden_Recenter_topo_20_crs_225_7_20m.shp'
# weatherstation
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/Ullstinden_Recenter_topo_20_crs_point_03-19-2025_1503_20m.shp'
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/multiple weatherstations/Ullstinden_Recenter_topo_20_crs_point_03-19-2025_1503_20m.shp'
#shapefile_path =  '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Ullstinden_20m/20m DEM original/weatherstations ullstinden 20m/neue stationen/Ullstinden_Recenter_topo_20_crs_point_03-31-2025_1924_20m.shp'

## toymountain 
#nc_file ='/Users/sarahvalent/Desktop/MASTER-THESIS/RESULT ANALYSIS/toymountain/PALM 20m/toy_mountain_smooth_inflow_20m_av_masked_M01.000.nc'
#averaged
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resoultiuon 20m DEM _averaged/toy_mountain_smooth_inflow_20m_topo_270_1_20m.shp'
#weather station
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM_ weatherstations/new with xy/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'
#one weather station
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM _ one weather station/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'
#shapefile_path = '/Users/sarahvalent/Desktop/MASTER-THESIS/Windows/Toymountain 20m DEM/20m resolution 20m DEM _ one weather station/with multipe 270/toy_mountain_smooth_inflow_20m_topo_point_03-19-2025_1503_20m.shp'

ftle_shapefile = FTLECalculator(shapefile_path)
ftle_netcdf = FTLECalculator(nc_file, is_netcdf=True)


#ftle_shapefile.compare_FTLE(ftle_netcdf)


ftle_shapefile_result = ftle_shapefile.compute_FTLE()
ftle_netcdf_result = ftle_netcdf.compute_FTLE()

def normalization_ftle(datafile):
    m =  np.mean(datafile)
    s = np.std(datafile)
    return (datafile - m)/s




normalized_nc_ftle = normalization_ftle(ftle_netcdf_result)
normalized_shape_ftle = normalization_ftle(ftle_shapefile_result)
print(np.std(ftle_shapefile_result))

def plot_FTLE(ftle_data, x_domain, y_domain, title, vmax):
    plt.figure(figsize=(8, 6))
    

    cf = plt.contourf(x_domain, y_domain, ftle_data, levels=250, cmap="seismic", vmax=vmax)
    
   
    cbar = plt.colorbar(cf)
    cbar.set_label("Normalized FTLE (Unit Variance)", fontsize=16)
    

    plt.title(title, fontsize=16)
    plt.xlabel("X-axis", fontsize=16)
    plt.ylabel("Y-axis", fontsize=16)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.show()

vmax = max(np.abs(normalized_nc_ftle).max(), np.abs(normalized_shape_ftle).max())

plot_FTLE(normalized_shape_ftle, ftle_shapefile.x_domain, ftle_shapefile.y_domain, "Windninja FTLE - reference", vmax)
plot_FTLE(normalized_nc_ftle, ftle_netcdf.x_domain, ftle_netcdf.y_domain, "Palm FTLE ",vmax)


