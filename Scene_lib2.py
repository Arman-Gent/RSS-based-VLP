import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.stats import qmc
import pandas as pd
from scipy.interpolate import RegularGridInterpolator



#----------------- General functions to work with the extended coordinate system
def augmented_coordinates(rotation,position):
    coord= np.eye(4)
    coord[:3,:3]=rotation
    coord[:3,3]=position
    return (coord)
def random_orientation():
    random_quaternion = np.random.randn(4)
    random_quaternion /= np.linalg.norm(random_quaternion)
    rotation_matrix = R.from_quat(random_quaternion).as_matrix()   
    return rotation_matrix
def get_position(U):
    return(U[:3,3])
def get_rotation(U):
    return(U[:3,:3])




#----------------- Implements the LightSource class, the PhotoDiode class as well as the lambertian radiation pattern
def radiation_lambertian(vector,m=1, orientation = np.eye(3),P=1,type=0):
    normal=orientation[:,2]
    distance = np.linalg.norm(vector)
    if distance ==0:
        return float('inf')
    direction = vector/distance
    normal= normal/ np.linalg.norm(normal)
    cos_theta = np.max([np.dot(vector, normal)/distance,0])
    I= (m+1)*P/(2*np.pi*distance**2)*cos_theta**m
    return(I)
class LightSource:
    def __init__(s, position= np.zeros(3),rotation= np.eye(3),m=1,P=1,type=0):
        s.pos=np.array(position)
        s.rot=np.array(rotation)
        s.m=m
        s.type=type
        s.P=P
        s.U= augmented_coordinates(rotation=s.rot,position=s.pos)
        s.normal= s.rot[:,2]/np.linalg.norm(s.rot[:,2])
        s.radiation_pattern= radiation_lambertian
    def radiation_intensity(s,position):
        vector = position - s.pos
        vector = np.dot(s.rot.T,vector)
        return(s.radiation_pattern(vector=vector,m=s.m,P=s.P,type=s.type))
    def __str__(s):
        rot_str = '\n'.join([f'[{float(row[0]):.2f}, {float(row[1]):.2f}, {float(row[2]):.2f}]' for row in s.rot])
        return f'Pos: {s.pos}\nRot:\n{rot_str}\nm= {s.m}\nP= {s.P} \n'
class PhotoDiode:
    def __init__(s,position= np.zeros(3), rotation = np.eye(3),AR=1):
        s.AR= AR
        s.pos=np.array(position)
        s.rot=np.array(rotation)
        s.normal= s.rot[:,2]/np.linalg.norm(s.rot[:,2])
        s.U= augmented_coordinates(rotation=s.rot,position=s.pos)
    def __str__(s):
        rot_str = '\n'.join([f'[{float(row[0]):.2f}, {float(row[1]):.2f}, {float(row[2]):.2f}]' for row in s.rot])
        return f'Pos: {s.pos}\nRot:\n{rot_str}\nAR= {s.AR} \n'
    

#----------------- Looks at the normalized measurement files of LEDs from Nobby, then it interpolates them in the radiation_measured radiation pattern

file_path = './Nobby_data/'
measured_LEDs= []
for i in range(1,5):
    xl= file_path+'LED'+str(i)+'arman.xlsx'
    df = pd.read_excel(xl, header=None, engine='openpyxl')
    led_pattern=df.to_numpy()
    measured_LEDs.append(led_pattern)


def radiation_measured(vector,P=1,type=0,orientation = np.eye(3),m=1):
    i_bar= orientation[:,0]
    j_bar= orientation[:,1]
    k_bar= orientation[:,2]
    i_bar/=np.linalg.norm(i_bar)
    j_bar/=np.linalg.norm(j_bar)
    k_bar/=np.linalg.norm(k_bar)
    distance = np.linalg.norm(vector)
    if distance ==0:
        return float('inf')
    direction = vector/distance
    cos_theta=np.dot(direction,k_bar)
    theta= np.arccos(cos_theta)
    i_component= np.dot(direction,i_bar)
    j_component= np.dot(direction,j_bar)
    phi = np.arctan2(j_component,i_component)
    if phi < 0:
        phi += 2 * np.pi
    rad_pattern=measured_LEDs[type]
    theta_grid = np.linspace(0, np.pi, rad_pattern.shape[0])
    phi_grid = np.linspace(0, 2 * np.pi, rad_pattern.shape[1], endpoint=False)
    

    interpolator = RegularGridInterpolator((theta_grid, phi_grid), rad_pattern, bounds_error=False, fill_value=None)

    intensity = interpolator((theta, phi))
    
    return intensity*P*distance**(-2)


# The Scene class implementation: make light sources and diodes, can 3d print the scene and look at light intensities for xy and xz planes. Can also be used to generate data. Sven light sources available, also photodiodes can be placed according to halton sampling
class Scene:
    def __init__(s, light_sources=None, photo_diodes=None):
        s.light_sources = [] if light_sources is None else light_sources
        s.photo_diodes = [] if photo_diodes is None else photo_diodes
    def make_led(s, position= np.zeros(3), rotation= np.eye(3), m=1,P=1):
        s.light_sources.append(LightSource(position=np.array(position),rotation=np.array(rotation),m=m,P=P))
    def make_diode(s, position= np.zeros(3),rotation = np.eye(3), AR=1):
        s.photo_diodes.append(PhotoDiode(position=np.array(position),rotation=np.array(rotation),AR=AR))
    def give_distance(s, led, diode):
        return(np.linalg.norm(led.pos- diode.pos))
    
    def RSS(s,led,diode,noise_std=0.,seed=None):
        rng= np.random.default_rng(seed=seed)
        intensity= led.radiation_intensity(diode.pos)
        vector = diode.pos-led.pos
        distance=np.linalg.norm(vector)
        cos_diode= np.max([np.dot(-vector, diode.normal)/distance,0])
        noise= rng.normal(loc=0., scale=noise_std)
        return(diode.AR*cos_diode*intensity+noise)

    def __str__(s):

        scene_str = "-------------- Light Sources --------------\n"
        for i, led in enumerate(s.light_sources):
            scene_str += f"LED{i}:\n{str(led)}\n"
        
        scene_str += "-------------- Photo Diodes --------------\n"
        for i, diode in enumerate(s.photo_diodes):
            scene_str += f"Diode{i}:\n{str(diode)}\n"
        return scene_str
        
    def xz(s,led=None,x_values=np.linspace(-4., 4., 100),z_values = np.linspace(0, 5, 100),y=0,diode_rot=np.eye(3),noise_std=0.,seed=None):
        X, Z = np.meshgrid(x_values, z_values)
        rng=np.random.default_rng(seed=seed)
        led = s.light_sources[0] if led is None else led
        intensity_values = np.zeros_like(X)
        diode=PhotoDiode(rotation=diode_rot)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                diode.pos = np.array([X[i, j], y, Z[i,j]])
                intensity_values[i, j] =s.RSS(led=led,diode=diode,noise_std=noise_std,seed=rng.integers(1e6))
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Z, np.log(intensity_values), levels=50, cmap='inferno')
        plt.colorbar(label='Intensity')
        plt.title('LED Radiation Pattern')
        plt.xlabel('X Position')
        plt.ylabel('Z Position')
        plt.show()
    def xy(s,led=None,x_values=np.linspace(-4., 4., 100),y_values = np.linspace(-4., 4., 100),z=0,diode_rot=np.eye(3),noise_std=0.,seed=None):
        X, Y = np.meshgrid(x_values, y_values)
        rng=np.random.default_rng(seed=seed)
        led = s.light_sources[0] if led is None else led
        intensity_values = np.zeros_like(X)
        diode=PhotoDiode(rotation=diode_rot)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                diode.pos = np.array([X[i, j], Y[i, j], z])
                intensity_values[i, j] =s.RSS(led=led,diode=diode,noise_std=noise_std,seed=rng.integers(1e6))
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, np.log(intensity_values), levels=50, cmap='inferno')
        plt.colorbar(label='Intensity')
        plt.title('LED Radiation Pattern')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()
    def plot_3d_scene(s, xlim=(-4., 4.), ylim=(-4., 4.), zlim=(0, 5)):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        for i, led in enumerate(s.light_sources):
            color='r'
            if led.radiation_pattern == radiation_measured:
                color='g'
            ax.scatter(led.pos[0], led.pos[1], led.pos[2], c=color, label=f'LED{i}')
            ax.quiver(led.pos[0], led.pos[1], led.pos[2], 
                    led.normal[0], led.normal[1], led.normal[2], 
                    length=0.3, color=color, arrow_length_ratio=0.1, linewidth=0.5)
        for i, diode in enumerate(s.photo_diodes):
            ax.scatter(diode.pos[0], diode.pos[1], diode.pos[2], c='b', label=f'Diode{i}')
            ax.quiver(diode.pos[0], diode.pos[1], diode.pos[2], 
                    diode.normal[0], diode.normal[1], diode.normal[2], 
                    length=0.3, color='b', arrow_length_ratio=0.1, linewidth=0.5)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title('Scene Configuration')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        #ax.legend()
        plt.show()
    def generate_rss_table(s,noise_std=0.,seed=None):
        num_diodes = len(s.photo_diodes)
        num_leds = len(s.light_sources)
        rng=np.random.default_rng(seed=seed)
        rss_df = pd.DataFrame(index=[f'Diode{i}' for i in range(num_diodes)], 
                            columns=[f'LED{j}' for j in range(num_leds)])
        for i, diode in enumerate(s.photo_diodes):
            for j, led in enumerate(s.light_sources):
                rss_value = s.RSS(led, diode,noise_std=noise_std,seed=rng.integers(1e6))
                rss_df.at[f'Diode{i}', f'LED{j}'] = rss_value
        return rss_df
    def generate_rss_table_with_diode_coords(s,noise_std=0.,seed=None):
        num_diodes = len(s.photo_diodes)
        num_leds = len(s.light_sources)
        rng=np.random.default_rng(seed=seed)
        columns = [f'RSS{j}' for j in range(num_leds)] + ['X', 'Y', 'Z']
        rss_df = pd.DataFrame(index=[f'Diode{i}' for i in range(num_diodes)], columns=columns)
        for i, diode in enumerate(s.photo_diodes):
            rss_df.at[f'Diode{i}', 'X'] = diode.pos[0]
            rss_df.at[f'Diode{i}', 'Y'] = diode.pos[1]
            rss_df.at[f'Diode{i}', 'Z'] = diode.pos[2]
            for j, led in enumerate(s.light_sources):
                rss_value = s.RSS(led, diode,noise_std=noise_std,seed=rng.integers(1e6))
                rss_df.at[f'Diode{i}', f'RSS{j}'] = rss_value
        return rss_df
    def generate_sven_lights(self,s=2,h=5, orientation=np.array([[0,1,0],[1,0,0],[0,0,-1]]),use_measured_data=False):
        LED_positions= [[0,0,h],[-s,-s,h],[-s,s,h],[s,-s,h],[s,s,h]]
        for i in LED_positions:
            self.make_led(position=i,rotation=orientation,m=1,P=1)
        if use_measured_data==True:
            for i in range(len(LED_positions)):
                self.light_sources[-i-1].radiation_pattern= radiation_measured
                self.light_sources[-i-1].type= i % 4
    def generate_diode_plane(s,plane_boundary=[[-4,4],[-4,4]],n=4,h=0,orientation=np.eye(3)):
        x_locations= np.linspace(plane_boundary[0][0],plane_boundary[0][1],n)
        y_locations= np.linspace(plane_boundary[1][0],plane_boundary[1][1],n)
        X,Y = np.meshgrid(x_locations,y_locations)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                s.make_diode(position=np.array([X[i,j],Y[i,j],h]),rotation=orientation)
    def generate_diode_volume(s,volume_boundary=[[-4,4],[-4,4],[0,5]],n=4,orientation=np.eye(3)):
        x_locations= np.linspace(volume_boundary[0][0],volume_boundary[0][1],n)
        y_locations= np.linspace(volume_boundary[1][0],volume_boundary[1][1],n)
        z_locations= np.linspace(volume_boundary[2][0],volume_boundary[2][1],n)
        X,Y,Z = np.meshgrid(x_locations,y_locations,z_locations)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    s.make_diode(position=np.array([X[i,j,k],Y[i,j,k],Z[i,j,k]]),rotation=orientation)
    def generate_diode_halton(s,plane_boundary=[[-4,4],[-4,4]],n=50,h=0,orientation=np.eye(3)):
        sampler = qmc.Halton(d=2,scramble = False)
        sampler.fast_forward(1)
        sample = sampler.random(n)
        sample = qmc.scale(sample,l_bounds=[plane_boundary[0][0],plane_boundary[1][0]],u_bounds=[plane_boundary[0][1],plane_boundary[1][1]])
        for i in sample:
            s.make_diode(position=np.array([i[0],i[1],h]),rotation=orientation)

    def generate_diode_halton_volume(s, volume_boundary=[[-4,4],[-4,4],[0,5]], n=50, orientation=np.eye(3), seed=None):
        sampler = qmc.Halton(d=3, scramble=False)
        sampler.fast_forward(1)
        if seed is not None:
            sampler.reset()
            sampler.fast_forward(seed*n)

        sample = sampler.random(n)
        sample = qmc.scale(sample, 
                        l_bounds=[volume_boundary[0][0], volume_boundary[1][0], volume_boundary[2][0]], 
                        u_bounds=[volume_boundary[0][1], volume_boundary[1][1], volume_boundary[2][1]])
        
        for i in sample:
            s.make_diode(position=np.array([i[0], i[1], i[2]]), rotation=orientation)

    def generate_rss_table_with_diode_coords_and_distances(s, noise_std=0., seed=None):

        num_diodes = len(s.photo_diodes)
        num_leds = len(s.light_sources)
        rng = np.random.default_rng(seed=seed)
        
        # Create column names for RSS and distances
        rss_columns = [f'RSS{j}' for j in range(num_leds)]
        distance_columns = [f'D{j}' for j in range(num_leds)]
        coord_columns = ['X', 'Y', 'Z']
        columns = rss_columns + distance_columns + coord_columns

        # Initialize the DataFrame
        rss_df = pd.DataFrame(index=[f'Diode{i}' for i in range(num_diodes)], columns=columns)

        for i, diode in enumerate(s.photo_diodes):
            # Add diode coordinates
            rss_df.at[f'Diode{i}', 'X'] = diode.pos[0]
            rss_df.at[f'Diode{i}', 'Y'] = diode.pos[1]
            rss_df.at[f'Diode{i}', 'Z'] = diode.pos[2]

            for j, led in enumerate(s.light_sources):
                # Calculate RSS value
                rss_value = s.RSS(led, diode, noise_std=noise_std, seed=rng.integers(1e6))
                rss_df.at[f'Diode{i}', f'RSS{j}'] = rss_value

                # Calculate distance
                distance = s.give_distance(led, diode)
                rss_df.at[f'Diode{i}', f'D{j}'] = distance

        return rss_df


    def generate_rss_table_with_diode_coords_relative(s, noise_std=0., seed=None):
        num_diodes = len(s.photo_diodes)
        num_leds = len(s.light_sources)
        rng = np.random.default_rng(seed=seed)

        # Generate RSS data
        rss_df = pd.DataFrame(index=[f'Diode{i}' for i in range(num_diodes)], columns=[f'RSS{j}' for j in range(num_leds)])
        
        for i, diode in enumerate(s.photo_diodes):
            for j, led in enumerate(s.light_sources):
                rss_value = s.RSS(led, diode, noise_std=noise_std, seed=rng.integers(1e6))
                rss_df.at[f'Diode{i}', f'RSS{j}'] = rss_value

        # Compute all possible RSS ratios (RSS_i / RSS_j for all pairs i < j)
        relative_rss_columns = []
        for i in range(num_leds):
            for j in range(i + 1, num_leds):  # Ensure unique pairs (i < j)
                relative_rss_columns.append(f"RSS{i}/RSS{j}")

        relative_rss_df = pd.DataFrame(index=rss_df.index, columns=relative_rss_columns)

        for i, diode in enumerate(s.photo_diodes):
            rss_values = [rss_df.at[f'Diode{i}', f'RSS{j}'] for j in range(num_leds)]
            for idx, (p, q) in enumerate([(p, q) for p in range(num_leds) for q in range(p + 1, num_leds)]):
                # Compute RSS ratio: RSS_p / RSS_q
                relative_rss_df.at[f'Diode{i}', f"RSS{p}/RSS{q}"] = rss_values[p] / rss_values[q]

        # Add coordinates at the end of the table
        coords = pd.DataFrame(index=rss_df.index, columns=['X', 'Y', 'Z'])
        for i, diode in enumerate(s.photo_diodes):
            coords.at[f'Diode{i}', 'X'] = diode.pos[0]
            coords.at[f'Diode{i}', 'Y'] = diode.pos[1]
            coords.at[f'Diode{i}', 'Z'] = diode.pos[2]

        # Concatenate the final dataframe
        final_df = pd.concat([relative_rss_df, coords], axis=1)

        return final_df

