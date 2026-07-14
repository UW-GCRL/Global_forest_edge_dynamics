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
from scipy import stats
from PIL import ImageColor
from matplotlib.colors import rgb2hex
from matplotlib.colors import to_rgba
import matplotlib.colors as mcolors
import geopandas as gpd
import xarray as xr
import matplotlib as mpl
import matplotlib.patches as mpatches
from shapely.geometry import mapping
from matplotlib import gridspec
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import cartopy.io.shapereader as shpreader
warnings.filterwarnings('ignore')


def read_data(image_data, bx=3600, by=1400):
    dataset = gdal.Open(image_data)
    geotransform = dataset.GetGeoTransform()
    origin_x,origin_y = geotransform[0],geotransform[3]
    pixel_width, pixel_height = geotransform[1],geotransform[5]

    width, height = dataset.RasterXSize, dataset.RasterYSize
    # Decimated read (GDAL resamples on read). The rasters are 36000x14000; drawn in small
    # multi-panel maps, a full-res pcolormesh renders each cell sub-pixel and the sparse
    # forest-edge data vanishes (and takes ~80 min). Reading at ~display resolution keeps
    # ~1 cell per output pixel -> dense like the original, and fast. Lossless at print dpi.
    data = dataset.GetRasterBand(1).ReadAsArray(buf_xsize=bx, buf_ysize=by)
    lon = origin_x + (pixel_width  * width)  * (np.arange(bx)/bx)
    lat = origin_y + (pixel_height * height) * (np.arange(by)/by)
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

data_path = "/scratch/fji7/Forest_edge_mapping_2024_11_14/1_data/"
images = ["stable.tif","ii.tif","id.tif","di.tif","dd.tif"]
images = [f"{data_path}{image}" for image in images]


#############################################
vims = [12, 4, 4, 4, 4]
cmaps = ['YlGnBu', 'YlOrBr', 'YlOrBr', 'YlOrBr', 'YlOrBr']
titles = ["a. Stable Forest Edge", "b. Forest Edge Increase due to Forest Extent Gain", 
          "c. Forest Edge Increase due to Forest Extent Lost", "d. Forest Edge Decrease due to Forest Extent Gain", 
          "e. Forest Edge Decrease due to Forest Extent Lost"]
levs = [np.arange(0,12.1,4), np.arange(0,4.1,2), np.arange(0,4.1,2), np.arange(0,4.1,2), np.arange(0,4.1,2)]
#****************************************************#
mm = 1/25.4
import matplotlib.font_manager as fm
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
fig = plt.figure(figsize = (230*mm, 177*mm))   # oversized canvas; tight-crop lands ~180 mm wide
gs = gridspec.GridSpec(9, 20)
plt.subplots_adjust(hspace =0,wspace =0.5)

ax1 = plt.subplot(gs[0:3, 5:15],projection = ccrs.Robinson(central_longitude=0.0))
ax2 = plt.subplot(gs[3:6, 0:10],projection = ccrs.Robinson(central_longitude=0.0))
ax3 = plt.subplot(gs[3:6, 10:20],projection = ccrs.Robinson(central_longitude=0.0))
ax4 = plt.subplot(gs[6:9, 0:10],projection = ccrs.Robinson(central_longitude=0.0))
ax5 = plt.subplot(gs[6:9, 10:20],projection = ccrs.Robinson(central_longitude=0.0))

axes = [ax1, ax2, ax3, ax4, ax5]

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
    ax.text(-0.05,1.03, titles[idx], transform=ax.transAxes, fontsize = 9,fontweight='bold')
    ax.set_extent([-179.99, 179.99, -90, 90])

    _botlab = mtransforms.offset_copy(ax.transData, fig=fig, y=-4, units='points')
    for _lon, _lab in [(-180, '180°'), (-90, '90°W'), (0, '0°'), (90, '90°E'), (180, '180°')]:
        _lx, _ly = ax.projection.transform_point(_lon, -90, ccrs.PlateCarree())
        ax.text(_lx, _ly, _lab, transform=_botlab, ha='center', va='top', fontsize=6, color='k', zorder=8)

    data,lon,lat,geotransform = read_data(image)

    # data = zoom(data,0.1)                               ####
    # lat,lon = zoom(lat,0.1),zoom(lon,0.1)               ####

    lons, lats = np.meshgrid(lon, lat)
    data[data < 0] = np.nan
    # pcolormesh preserves spatially sparse forest-edge data (empty cells stay transparent);
    # imshow would average it away when downsampled to display resolution.
    p = ax.pcolormesh(lons,lats,data,transform=ccrs.PlateCarree(), cmap=cmaps[idx], vmin = 0, vmax = vims[idx], rasterized=True)

    cax, kw = mpl.colorbar.make_axes(ax, location='bottom', pad=0.06, shrink=0.2,anchor = (0.55,2.9))
    cbar = plt.colorbar(p, cax=cax, orientation='horizontal',ticks=levs[idx])
    cbar.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=6,pad = 0.1,length=1)
    cbar.set_label('Forest edge length (km)',fontsize = 7,labelpad=2)

out = '/scratch/fji7/Forest_edge_mapping_2024_11_14/2_exported_figures/Figure S1_Forest edge length'
plt.savefig(out + '.pdf', dpi=600, bbox_inches='tight')
# PNG preview generated from the PDF via pdftoppm (pcolormesh PNG re-rasterizes the mesh and hangs).
