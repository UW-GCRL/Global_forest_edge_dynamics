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
import matplotlib.transforms as mtransforms
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
    lat = origin_y + pixel_height * np.arange(height)
    
    data = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    # data[data == data[0][0]] = np.nan
    return data,lon,lat,geotransform

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

def country_extraction(global_data,lat,lon,country_shp):
    file = xr.DataArray(global_data, coords=[('lat', lat), ('lon', lon), ('channel', [1, 2, 3, 4])])
    ds = xr.Dataset({'data': file})
    ds.rio.write_crs("EPSG:4326", inplace=True)
    ds = ds.rename({'lon': 'x'})
    ds = ds.rename({'lat': 'y'})
    clipped = ds.rio.clip(country_shp.geometry.apply(mapping), country_shp.crs, drop=False)
    region_data = clipped.variables['data']
    return region_data
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

#**********************************************************************************************
edge_file = '/scratch/fji7/Forest_edge_mapping_2024_11_14/1_data/2000_2020_edge_diff_per_range.tif'
area_file = '/scratch/fji7/Forest_edge_mapping_2024_11_14/1_data/2000_2020_area_diff_per_range.tif'

data1,lon1,lat1,transform1 = read_data(edge_file)
data2,lon2,lat2,transform2 = read_data(area_file)

# data1, data2 = zoom(data1,0.01),zoom(data2,0.01)                                                             ##########
# lat1,lon1,lat2,lon2 = zoom(lat1,0.01),zoom(lon1,0.01),zoom(lat2,0.01),zoom(lon2,0.01)                        ##########

mask = (data1>= -1) & (data1 <= 1) & (data2 >= -1) & (data2 <= 1)
data1[data1 < -1] = -1
data2[data2 < -1] = -1
data1[data1 > 1] = 1
data2[data2 > 1] = 1

print(np.nanmax(data1))
print(np.nanmin(data1))
print(np.nanmax(data2))
print(np.nanmin(data2))

data1_clipped = np.clip(data1, -0.2, 0.2)
data2_clipped = np.clip(data2, -0.2, 0.2)
data1 = None
data2 = None

data1_clipped_norm = (data1_clipped+0.2)/0.4
data2_clipped_norm = (data2_clipped+0.2)/0.4
data1_clipped = None
data2_clipped = None

print(np.nanmax(data1_clipped_norm))
print(np.nanmin(data1_clipped_norm))
print(np.nanmax(data2_clipped_norm))
print(np.nanmin(data2_clipped_norm))


n = 40
edge_indices = (data1_clipped_norm * (n - 1)).astype(int)
area_indices = (data2_clipped_norm * (n - 1)).astype(int)

bivariate_colors = np.empty((n, n, 4))
for i in range(n):
    for j in range(n):
        bivariate_colors[i, j] = custom_colormap(i, j, n)

# Vectorized bivariate lookup — identical result to the original per-pixel double loop
# (504 M px), but runs in seconds. Indices are clipped so masked / out-of-range pixels
# index safely; those pixels are then zeroed via the mask exactly as before.
ai = np.clip(area_indices, 0, n - 1)
ej = np.clip(edge_indices, 0, n - 1)
final_rgba = bivariate_colors[ai, ej].astype(float)
final_rgba[~mask] = 0
            
            
world_filepath = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
world = gpd.read_file(world_filepath)
countries = ['United States of America','China', 'Taiwan','Russia','Brazil','Canada']
selected_countries = world[world['SOVEREIGNT'].isin(countries)]
china = selected_countries[(selected_countries['SOVEREIGNT']=='China')|(selected_countries['SOVEREIGNT']=='Taiwan')]
us = selected_countries[(selected_countries['SOVEREIGNT']=='United States of America')&(selected_countries['TYPE']=='Country')]
russia = selected_countries[selected_countries['SOVEREIGNT']=='Russia']
brazil = selected_countries[selected_countries['SOVEREIGNT']=='Brazil']
canada = selected_countries[selected_countries['SOVEREIGNT']=='Canada']
contries_shp = [china, us, russia, brazil, canada]


start_t = datetime.datetime.now()
print('start:', start_t)

mm = 1/25.4                                   # mm -> inch helper (Nature specifies mm)

# Helvetica/Arial are not installed on the cluster; register TeX Gyre Heros (a
# metric-compatible Helvetica clone) by absolute path so any compute node uses it.
import matplotlib.font_manager as fm
_FONTDIR = '/scratch/fji7/0_pyenv/fonts/heros_otf'
for _f in ['texgyreheros-regular.otf', 'texgyreheros-bold.otf', 'texgyreheros-italic.otf']:
    fm.fontManager.addfont(f'{_FONTDIR}/{_f}')
_SANS = fm.FontProperties(fname=f'{_FONTDIR}/texgyreheros-regular.otf').get_name()

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': [_SANS, 'Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.default': 'regular',             # math digits/letters match the sans font
    'font.size':        7,                     # Nature hard cap = 7 pt (floor = 5 pt)
    'axes.titlesize':   7,
    'axes.labelsize':   7,
    'xtick.labelsize':  6,
    'ytick.labelsize':  6,
    'legend.fontsize':  6,
    'axes.linewidth':   0.5,
    'pdf.fonttype':    42,                     # embed text as editable text, not outlines
    'ps.fonttype':     42,
})
# Canvas oversized on purpose: bbox_inches='tight' crops the saved PDF to content,
# so this lands at ~177 x 223 mm (fits Nature's 180 x 225 mm envelope) with text at 5-7 pt.
fig = plt.figure(figsize = (211*mm, 211*mm))
gs = gridspec.GridSpec(21, 6)
plt.subplots_adjust(hspace =0,wspace =0.25)

####################### global data
ax = plt.subplot(gs[0:10, :],projection = ccrs.Robinson(central_longitude=0.0))

base_map(ax)
grd = ax.gridlines(draw_labels=True, xlocs=range(-180, 181, 90), ylocs=range(-60, 61, 30), color='gray',linestyle='--', linewidth=0.5, zorder=7)
# cartopy 0.18 mis-places longitude labels on the curved Robinson boundary (duplicated at
# the top and rotated along the gridlines). Turn its longitude labels off and add plain
# horizontal ones on the bottom edge only.
grd.top_labels = False
grd.ylabel_style = {'size': 6}                            # latitude labels: match the 6 pt longitude labels below
grd.xformatter = mtick.FuncFormatter(lambda v, pos: '')   # blank cartopy's longitude labels (0.18 keeps mis-placed ones otherwise)
# add plain horizontal longitude labels along the actual bottom edge of the map
_bx0, _bx1, _blat, _btop = ax.get_extent(ccrs.PlateCarree())
_botlab = mtransforms.offset_copy(ax.transData, fig=fig, y=-4, units='points')
for _lon, _lab in [(-180, '180°'), (-90, '90°W'), (0, '0°'), (90, '90°E'), (180, '180°')]:
    _x, _y = ax.projection.transform_point(_lon, _blat, ccrs.PlateCarree())
    ax.text(_x, _y, _lab, transform=_botlab, ha='center', va='top', fontsize=6, color='k', zorder=8)

ax.tick_params(axis='both',which='major',labelsize=6,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,
                bottom=True,left=True,top=False,right=False)
ax.spines['geo'].set_linewidth(0.7)
ax.text(-0.05,1.03, 'a. Global Forest Edge Dynamics', transform=ax.transAxes, fontsize = 9,fontweight='bold')
ax.imshow(final_rgba, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(),zorder = 3, rasterized=True)
china.plot(ax = ax,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
us.plot(ax = ax,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
russia.plot(ax = ax,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
brazil.plot(ax = ax,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
canada.plot(ax = ax,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)

axx = ax.inset_axes([0.08,0.18,0.25,0.25])
axx.set_aspect('equal', adjustable='box')

axx.imshow(bivariate_colors, origin='lower', extent=[0, 1, 0, 1], rasterized=True)

axx.tick_params(axis='both',which='major',bottom=False, left=False,top=False,right=False,labelleft = False, labelbottom = False)
axx.spines[['top', 'right','left','bottom']].set_visible(False)
axx.set_title('Forest Dynamics',fontsize = 6)
axx.set_xlabel('edge changes',fontsize = 6,labelpad = 8)
axx.set_ylabel('area changes',fontsize = 6,labelpad = 8)
axx.text(1,-0.05, '+20%', transform=axx.transAxes, fontsize = 6)
axx.text(-0.15,0.78, '+20%', transform=axx.transAxes, fontsize = 6,rotation= 90)
axx.text(-0.22,-0.22, '-20%', transform=axx.transAxes, fontsize = 6,rotation= 45)

####################### regional data
ax1 = plt.subplot(gs[11:16, 0:2],projection = ccrs.PlateCarree())
ax2 = plt.subplot(gs[11:16, 2:4],projection = ccrs.PlateCarree())
ax4 = plt.subplot(gs[11:16, 4:6],projection = ccrs.PlateCarree())
ax3 = plt.subplot(gs[16:21, 0:4],projection = ccrs.PlateCarree())
ax5 = plt.subplot(gs[16:21, 4:6],projection = ccrs.PlateCarree())

ax1.text(0,1.05, 'b. Forest Edge Dynamics in Different Countries', transform=ax1.transAxes, fontsize = 9,fontweight='bold')
ax1.text(0.02,0.9, '(b.1) China', transform=ax1.transAxes, fontsize = 6)
ax2.text(0.02,0.9, '(b.2) U.S. mainland', transform=ax2.transAxes, fontsize = 6)
ax4.text(0.02,0.9, '(b.3) Brazil', transform=ax4.transAxes, fontsize = 6)
ax3.text(0.02,0.9, '(b.4) Russia', transform=ax3.transAxes, fontsize = 6)
ax5.text(0.02,0.9, '(b.5) Canada', transform=ax5.transAxes, fontsize = 6)

d1 = country_extraction(final_rgba,lat1,lon1,china)
d2 = country_extraction(final_rgba,lat1,lon1,us)
d3 = country_extraction(final_rgba,lat1,lon1,russia)
d4 = country_extraction(final_rgba,lat1,lon1,brazil)
d5 = country_extraction(final_rgba,lat1,lon1,canada)

base_map(ax1)
base_map(ax2)
base_map(ax3)
base_map(ax4)
base_map(ax5)

ax1.imshow(d1, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(), zorder = 3, rasterized=True)
ax2.imshow(d2, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(), zorder = 3, rasterized=True)
ax3.imshow(d3, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(), zorder = 3, rasterized=True)
ax4.imshow(d4, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(), zorder = 3, rasterized=True)
ax5.imshow(d5, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(), zorder = 3, rasterized=True)

china.plot(ax = ax1,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
us.plot(ax = ax2,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
russia.plot(ax = ax3,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
brazil.plot(ax = ax4,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
canada.plot(ax = ax5,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)

ax1.set_extent([69, 137, 14, 50])
ax2.set_extent([-128, -60, 18, 54])
ax3.set_extent([20, 179, 40, 75])
ax4.set_extent([-90, -22, -31, 5])
ax5.set_extent([-145, -50, 30, 85])

ax1.set_xticks([80,100,120])
ax2.set_xticks([-120,-100,-80,-60])
ax3.set_xticks([30,60,90,120,150])
ax4.set_xticks([-80,-60,-40])
ax5.set_xticks([-140,-120,-100,-80, -60])

ax1.set_yticks([20,35,50])                 
ax2.set_yticks([30,50])
ax3.set_yticks([45,60,75])
ax4.set_yticks([-30,-15, 0])
ax5.set_yticks([40,60,80])

ax1.set_yticklabels([x.get_text() for x in ax1.get_yticklabels()],rotation=90, va='center')
ax2.set_yticklabels([x.get_text() for x in ax2.get_yticklabels()],rotation=90, va='center')
ax3.set_yticklabels([x.get_text() for x in ax3.get_yticklabels()],rotation=90, va='center')
ax4.set_yticklabels([x.get_text() for x in ax4.get_yticklabels()],rotation=90, va='center')
ax5.set_yticklabels([x.get_text() for x in ax5.get_yticklabels()],rotation=90, va='center')

ax1.xaxis.set_major_formatter(LongitudeFormatter()) 
ax2.xaxis.set_major_formatter(LongitudeFormatter()) 
ax3.xaxis.set_major_formatter(LongitudeFormatter()) 
ax4.xaxis.set_major_formatter(LongitudeFormatter()) 
ax5.xaxis.set_major_formatter(LongitudeFormatter()) 
                   
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax2.yaxis.set_major_formatter(LatitudeFormatter())
ax3.yaxis.set_major_formatter(LatitudeFormatter())
ax4.yaxis.set_major_formatter(LatitudeFormatter())
ax5.yaxis.set_major_formatter(LatitudeFormatter())

ax1.tick_params(axis='both',which='major',labelsize=6,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True, bottom=True,left=True,top=False,right=False)
ax2.tick_params(axis='both',which='major',labelsize=6,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True, bottom=True,left=True,top=False,right=False)
ax3.tick_params(axis='both',which='major',labelsize=6,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True, bottom=True,left=True,top=False,right=False)
ax4.tick_params(axis='both',which='major',labelsize=6,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True, bottom=True,left=True,top=False,right=False)
ax5.tick_params(axis='both',which='major',labelsize=6,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True, bottom=True,left=True,top=False,right=False)

ax1.spines['geo'].set_linewidth(0.7)
ax2.spines['geo'].set_linewidth(0.7)
ax3.spines['geo'].set_linewidth(0.7)
ax4.spines['geo'].set_linewidth(0.7)
ax5.spines['geo'].set_linewidth(0.7)

######### statistics
df = pd.read_csv('/scratch/fji7/Forest_edge_mapping_2024_11_14/1_data/processed_country_data_with_area.csv')
countries = ['United States of America','China', 'Taiwan','Russia','Brazil','Canada']
df = df[df['country'].isin(countries)]
df = df[['country','increase increase','increase decrease','decrease increase','decrease decrease']]

china_df = df[(df['country'] == 'China')|(df['country'] == 'Taiwan')]
china_df = pd.DataFrame(china_df.sum(axis = 0)).T
china_df['country'] = 'China'

other_df = df[(df['country'] == 'United States of America')|(df['country'] == 'Russia')|
              (df['country'] == 'Canada')|(df['country'] == 'Brazil')]
df = pd.concat([china_df,other_df], axis = 0)
var = True
for i in ['China', 'United States of America','Brazil', 'Russia','Canada']:
    temp = df[df['country'] == i]
    temp = temp[['increase increase','increase decrease','decrease increase','decrease decrease']]
    temp = temp.T
    temp.reset_index(inplace = True)
    temp.columns = ['dynamics','values']
    temp['country'] = 'U.S.' if i == 'United States of America' else i
    if var:
        final_df = temp
        var = False
    else:
        final_df = pd.concat([final_df,temp], axis = 0)    
final_df.reset_index(drop = True, inplace = True)

axes = inset_axes(ax, width="60%", height="35%", loc='lower left', bbox_to_anchor=(-0.02, -1.65, 1, 1),bbox_transform=ax.transAxes)
axes.text(-0.06,1.25, 'c. Forest Edge Dynamics Statistics', transform=axes.transAxes, fontsize = 9,fontweight='bold')
sns.barplot(x='country', y= 'values', hue = 'dynamics', ax = axes, data=final_df,saturation=0.9, errcolor='k',errwidth = 0.7,
            palette=[bivariate_colors[n-1][n-1],bivariate_colors[0][n-1], bivariate_colors[n-1][0], bivariate_colors[0][0]],capsize = 0.07,edgecolor="k",linewidth = 0.7)

axes.spines[['top', 'right']].set_visible(False)
axes.spines[['bottom','left']].set_linewidth(1)
axes.tick_params(axis='both',which='major',labelsize=6,direction='in',pad=5,bottom=True, left=True,top=False,right=False)
axes.set_xlabel('Countries',fontsize = 7,labelpad = 2)
axes.set_ylabel('Forest Edge (km)',fontsize = 7,labelpad = 2)
axes.ticklabel_format(style='scientific', scilimits=(0, 0), axis='y')
axes.yaxis.offsetText.set_visible(False)
axes.text(-0.03,1.05, r"$\times 10^7$", transform=axes.transAxes, fontsize = 6)

legend = axes.legend()
handles = legend.legendHandles
labels=['Forest edge increase due to forest extent gain', 'Forest edge increase due to forest extent lost',
        'Forest edge decrease due to forest extent gain','Forest edge decrease due to forest extent lost']
axes.legend(handles = handles, labels = labels, loc = 'lower right',fontsize=7,facecolor= 'none',edgecolor = 'none',bbox_to_anchor=(1.78, -0.06))
# Vector PDF (submission) + 300-dpi PNG (preview). Raster map layers embedded at 300 dpi;
# coastlines, country borders, text and axes stay vector and editable.
out = '/scratch/fji7/Forest_edge_mapping_2024_11_14/2_exported_figures/Figure 3_Forest edge dynamics'
plt.savefig(out + '.pdf', dpi=600, bbox_inches='tight')
plt.savefig(out + '.png', dpi=600, bbox_inches='tight')

end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('end:', end_t)
print('total:',elapsed_sec/60, 'min')