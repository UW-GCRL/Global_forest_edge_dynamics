import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cf
from scipy.ndimage import zoom
from scipy.signal import savgol_filter
import matplotlib.ticker as mticker
import matplotlib.ticker as mtick
import matplotlib as mpl
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime
from PIL import ImageColor
from matplotlib.colors import rgb2hex
from matplotlib.colors import to_rgba
import matplotlib.colors as mcolors
import geopandas as gpd
import xarray as xr
from shapely.geometry import mapping
import rioxarray
from matplotlib import gridspec
import rasterio
import cartopy.io.shapereader as shpreader
warnings.filterwarnings('ignore')


def read_data(image_data):
    dataset = gdal.Open(image_data)
    geotransform = dataset.GetGeoTransform()
    origin_x,origin_y = geotransform[0],geotransform[3]
    pixel_width, pixel_height = geotransform[1],geotransform[5]
    
    width, height = dataset.RasterXSize, dataset.RasterYSize
    lon = origin_x + pixel_width * np.arange(width)
    lat = origin_y - pixel_height * np.arange(height)
    
    num_bands = dataset.RasterCount
    bands_data = []

    for i in range(1, num_bands + 1):
        band = dataset.GetRasterBand(i)
        band_data = band.ReadAsArray()
        bands_data.append(band_data)
        
    data = np.stack(bands_data, axis=-1)
    dataset = None
    return data,lon,lat,geotransform

def base_map(ax):
    states_provinces = cf.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',
                                              scale='50m',facecolor='none')
    ax.add_feature(cf.LAND,alpha=0.1)
    ax.add_feature(cf.BORDERS, linestyle='--',lw=0.4, alpha=0.5)
    ax.add_feature(cf.LAKES, alpha=0.5)
    ax.add_feature(cf.OCEAN,alpha=0.1,zorder = 2)
    ax.add_feature(cf.COASTLINE,lw=0.4)
    ax.add_feature(cf.RIVERS,lw=0.2)
    ax.add_feature(states_provinces,lw=0.2,edgecolor='gray')
    return

# def custom_colormap(i, j, n):
#     # Normalize the indices to the range [0, 1]
#     x = i / (n - 1)
#     y = j / (n - 1)
#     # Compute the distance from the center
#     distance = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
#     # Define the color components based on distance from center
#     g = np.clip(distance + (x - 0.5), 0, 1)
#     r = np.clip(distance + (y - 0.5), 0, 1)
#     b = np.clip(1 - distance, 0, 1)
#     return (r, g, b, 1)

# def custom_colormap(i, j, n):
#     x = i / (n - 1)
#     y = j / (n - 1)
#     c00 = (0.0, 0.0, 1.0)    # bright blue
#     c10 = (1.0, 0.0, 0.0)    # red
#     c01 = (0.0, 1.0, 0.0)    # green
#     c11 = (1.0, 1.0, 0.0)    # orange
#     bc_r = c00[0]*(1 - x)*(1 - y) + c10[0]*x*(1 - y) + c01[0]*(1 - x)*y + c11[0]*x*y
#     bc_g = c00[1]*(1 - x)*(1 - y) + c10[1]*x*(1 - y) + c01[1]*(1 - x)*y + c11[1]*x*y
#     bc_b = c00[2]*(1 - x)*(1 - y) + c10[2]*x*(1 - y) + c01[2]*(1 - x)*y + c11[2]*x*y
#     bc_color = (bc_r, bc_g, bc_b)
#     dx = x - 0.5
#     dy = y - 0.5
#     r = np.sqrt(dx*dx + dy*dy)
#     r_max = np.sqrt(0.5**2 + 0.5**2)  # ~0.707
#     t = min(1.0, r / r_max)
#     center_color = (0.85, 0.85, 0.85)
#     r_out = center_color[0]*(1 - t) + bc_color[0]*t
#     g_out = center_color[1]*(1 - t) + bc_color[1]*t
#     b_out = center_color[2]*(1 - t) + bc_color[2]*t
#     return (r_out, g_out, b_out, 1.0)

def custom_colormap(i, j, n):
    x = i / (n - 1)
    y = j / (n - 1)
    c00 = (0.0, 0.0, 1.0)    # bright blue
    c10 = (0.0, 1.0, 0.0)    # green
    c01 = (1.0, 0.0, 0.0)    # red
    c11 = (1.0, 1.0, 0.0)    # orange
    bc_r = c00[0]*(1 - x)*(1 - y) + c10[0]*x*(1 - y) + c01[0]*(1 - x)*y + c11[0]*x*y
    bc_g = c00[1]*(1 - x)*(1 - y) + c10[1]*x*(1 - y) + c01[1]*(1 - x)*y + c11[1]*x*y
    bc_b = c00[2]*(1 - x)*(1 - y) + c10[2]*x*(1 - y) + c01[2]*(1 - x)*y + c11[2]*x*y
    bc_color = (bc_r, bc_g, bc_b)
    dx = x - 0.5
    dy = y - 0.5
    r = np.sqrt(dx*dx + dy*dy)
    r_max = np.sqrt(0.5**2 + 0.5**2)  # ~0.707
    t = min(1.0, r / r_max)
    center_color = (0.85, 0.85, 0.85)
    r_out = center_color[0]*(1 - t) + bc_color[0]*t
    g_out = center_color[1]*(1 - t) + bc_color[1]*t
    b_out = center_color[2]*(1 - t) + bc_color[2]*t
    return (r_out, g_out, b_out, 1.0)

n = 40
bivariate_colors = np.empty((n, n, 4))
for i in range(n):
    for j in range(n):
        bivariate_colors[i, j] = custom_colormap(i, j, n)


#**********************************        
# data_path = "/mnt/cephfs/scratch/groups/chen_group/FujiangJi/Forest_edge_mapping_2024_11_14/absolute_and_realtive_color_array/"
# data_path = "/mnt/cephfs/scratch/groups/chen_group/FujiangJi/Forest_edge_mapping_2024_11_14/absolute_and_realtive_color_array/new_absolute_array/"
data_path = "/mnt/cephfs/scratch/groups/chen_group/FujiangJi/Forest_edge_mapping_2024_11_14/absolute_and_realtive_color_array/New_color/"

# images = ["absolute_color_array_2000_2005.tif", "absolute_color_array_2005_2010.tif",
#           "absolute_color_array_2010_2015.tif", "absolute_color_array_2015_2020.tif"]
# images = ["absolute_color_array_2000_2005_02_2.tif", "absolute_color_array_2005_2010_02_2.tif",
#           "absolute_color_array_2010_2015_02_2.tif", "absolute_color_array_2015_2020_02_2.tif"]

images = ["New_color_absolute_color_array_2000_2005_01_05.tif", "New_color_absolute_color_array_2005_2010_01_05.tif",
          "New_color_absolute_color_array_2010_2015_01_05.tif", "New_color_absolute_color_array_2015_2020_01_05.tif"]

images = [f"{data_path}{image}" for image in images]


mm = 1/25.4
import matplotlib.font_manager as fm
import matplotlib.transforms as mtransforms
_FONTDIR = '/scratch/fji7/0_pyenv/fonts/heros_otf'
for _f in ['texgyreheros-regular.otf', 'texgyreheros-bold.otf', 'texgyreheros-italic.otf']:
    fm.fontManager.addfont(f'{_FONTDIR}/{_f}')
_SANS = fm.FontProperties(fname=f'{_FONTDIR}/texgyreheros-regular.otf').get_name()
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': [_SANS, 'Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.default': 'regular',
    'font.size':        7, 'axes.titlesize': 7, 'axes.labelsize': 7,
    'xtick.labelsize':  6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth':   0.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})
fig = plt.figure(figsize = (230*mm, 106*mm))   # oversized canvas; tight-crop lands ~180 mm wide
gs = gridspec.GridSpec(10, 20)
plt.subplots_adjust(hspace =5,wspace =0.5)

titles = ["a. Global Forest Edge Dynamics (Absolute Difference, 2000 - 2005)", 
          "b. Global Forest Edge Dynamics (Absolute Difference, 2005 - 2010)", 
          "c. Global Forest Edge Dynamics (Absolute Difference, 2010 - 2015)",
          "d. Global Forest Edge Dynamics (Absolute Difference, 2015 - 2020)"]

ax1 = plt.subplot(gs[0:5, 0:10],projection = ccrs.Robinson(central_longitude=0.0))
ax2 = plt.subplot(gs[0:5, 10:20],projection = ccrs.Robinson(central_longitude=0.0))
ax3 = plt.subplot(gs[5:10, 0:10],projection = ccrs.Robinson(central_longitude=0.0))
ax4 = plt.subplot(gs[5:10, 10:20],projection = ccrs.Robinson(central_longitude=0.0))


axes = [ax1, ax2, ax3, ax4]

for idx, image in enumerate(images):
    print(idx, image)
    ax = axes[idx]
    base_map(ax)

    grd = ax.gridlines(draw_labels=True, xlocs=range(-180, 181, 90), ylocs=range(-60, 61, 30), color='gray',linestyle='--', linewidth=0.5, zorder=7)
    grd.top_labels = False
    grd.ylabel_style = {'size': 6}
    grd.xformatter = mtick.FuncFormatter(lambda v, pos: '')   # blank cartopy longitude labels; add horizontal ones below

    ax.tick_params(axis='both',which='major',labelsize=6,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,
                    bottom=True,left=True,top=False,right=False)
    ax.spines['geo'].set_linewidth(0.7)
    ax.text(-0.05,1.03, titles[idx], transform=ax.transAxes, fontsize = 7,fontweight='bold')

    data,lon,lat,geotransform = read_data(image)
    ax.imshow(data,extent = [lon.min(), lon.max(), lat.min(), lat.max()], transform=ccrs.PlateCarree(), rasterized=True)

    _bx0, _bx1, _blat, _btop = ax.get_extent(ccrs.PlateCarree())
    _botlab = mtransforms.offset_copy(ax.transData, fig=fig, y=-4, units='points')
    for _lon, _lab in [(-180, '180°'), (-90, '90°W'), (0, '0°'), (90, '90°E'), (180, '180°')]:
        _lx, _ly = ax.projection.transform_point(_lon, _blat, ccrs.PlateCarree())
        ax.text(_lx, _ly, _lab, transform=_botlab, ha='center', va='top', fontsize=6, color='k', zorder=8)
    
    
    axx = ax.inset_axes([0.09,0.17,0.25,0.25])
    axx.set_aspect('equal', adjustable='box')

    axx.imshow(bivariate_colors, origin='lower', extent=[0, 1, 0, 1])
    axx.tick_params(axis='both',which='major',bottom=False, left=False,top=False,right=False,labelleft = False, labelbottom = False)
    axx.spines[['top', 'right','left','bottom']].set_visible(False)
    axx.set_xlabel('edge changes',fontsize = 5,labelpad = 8)
    axx.set_ylabel('area changes',fontsize = 5,labelpad = 8)
    axx.text(1,-0.12, '+0.5 $km$', transform=axx.transAxes, fontsize = 5)
    axx.text(0,-0.12, '-0.5 $km$', transform=axx.transAxes, fontsize = 5)
    axx.text(-0.18,0.9, '+0.1 $km^2$', transform=axx.transAxes, fontsize = 5,rotation= 90)
    axx.text(-0.18,-0.1, '-0.1 $km^2$', transform=axx.transAxes, fontsize = 5,rotation= 90)

out = '/scratch/fji7/Forest_edge_mapping_2024_11_14/2_exported_figures/Figure S3_Forest edge dynamics_absolute difference'
plt.savefig(out + '.pdf', dpi=600, bbox_inches='tight')
plt.savefig(out + '.png', dpi=600, bbox_inches='tight')