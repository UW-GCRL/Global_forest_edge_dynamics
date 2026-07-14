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
import cartopy.io.shapereader as shpreader
warnings.filterwarnings('ignore')


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
        
#****************************************************************************************************************************************************************************************************************************************************       
df = pd.read_csv("/scratch/fji7/Forest_edge_mapping_2024_11_14/1_data/processed_country_data_with_area.csv")
world_filepath = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
reader = shpreader.Reader(world_filepath)
countries = [country.attributes['NAME'] for country in reader.records()]
continents = [country.attributes['CONTINENT'] for country in reader.records()]
country_to_continent = dict(zip(countries, continents))
df['continent'] = df['country'].map(country_to_continent)

colors = {'Asia': 'red','Africa': 'green','North America': 'blue','South America': 'yellow','Europe': 'purple','Oceania': 'cyan'}
df['Edge Change'] = df['Edge Change'].astype(float)
df['Area Change'] = df['Area Change'].astype(float)
df = df.dropna(subset=['Edge Change', 'Area Change'])

edge_change_threshold = 3 * df['Edge Change'].std()
area_change_threshold = 3 * df['Area Change'].std()
subset_df = df[(abs(df['Edge Change']) <= edge_change_threshold)& (abs(df['Area Change']) <= area_change_threshold)]

#****************************************************************************************************************************************************************************************************************************************************       
selected_countries = ["Russia", "Canada", "Brazil", "Laos", "Liberia", "India"]
years = [2000, 2005, 2010, 2015, 2020]

start_var = True
for year in years[:-1]:
    data = pd.read_csv(f"/scratch/fji7/Forest_edge_mapping_2024_11_14/1_data/country_data_with_area_{year}_{year+5}.csv")

    data = data[data['Country'].isin(selected_countries)]
    data = data[["Country", "Increase Increase Edge Length (KM)", "Increase Decrease Edge Length (KM)",
                 "Decrease Increase Edge Length (KM)", "Decrease Decrease Edge Length (KM)"]]
    data["time_period"] = f"{year}~{year+5}"
    if start_var:
        final_data = data
        start_var = False
    else:
        final_data = pd.concat([final_data, data], axis = 0)
final_data.reset_index(drop = True, inplace = True)

length_type = ["Increase Increase Edge Length (KM)", "Increase Decrease Edge Length (KM)", 
               "Decrease Increase Edge Length (KM)", "Decrease Decrease Edge Length (KM)"]
len_type = {"Increase Increase Edge Length (KM)":'Forest edge increase due to forest extent gain', 
            "Increase Decrease Edge Length (KM)":'Forest edge increase due to forest extent lost', 
            "Decrease Increase Edge Length (KM)":'Forest edge decrease due to forest extent gain', 
            "Decrease Decrease Edge Length (KM)":'Forest edge decrease due to forest extent lost'}

var = True
for length in length_type:
    temp = final_data[["Country", "time_period", length]]
    temp.rename(columns={length:'length'},inplace = True)
    temp["len_type"] = len_type[length]
    if var:
        final_df = temp
        var = False
    else:
        final_df = pd.concat([final_df, temp], axis = 0)
final_df.reset_index(drop = True, inplace = True)

#****************************************************************************************************************************************************************************************************************************************************       
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
# so this lands at ~180 x 85 mm (2-column width) with text at 5-7 pt.
fig = plt.figure(figsize = (209*mm, 77*mm))
gs = gridspec.GridSpec(21, 23)
plt.subplots_adjust(hspace =0,wspace =0.25)

ax = plt.subplot(gs[:, 0:8])
ax1 = plt.subplot(gs[0:9, 9:13])
ax2 = plt.subplot(gs[0:9, 14:18])
ax3 = plt.subplot(gs[0:9, 19:23])
ax4 = plt.subplot(gs[12:, 9:13])
ax5 = plt.subplot(gs[12:, 14:18])
ax6 = plt.subplot(gs[12:, 19:23])

#********
min_size = 8
max_size = 250

min_area = df['forest edge 2000'].min()
max_area = df['forest edge 2000'].max()
bubble_sizes = ((df['forest edge 2000'] - min_area) / (max_area - min_area) * (max_size - min_size) + min_size)

for continent, color in colors.items():
    subset = subset_df[subset_df['continent'] == continent]
    subset_bubble_sizes = bubble_sizes[subset.index]
    ax.scatter(subset['Edge Change'], subset['Area Change'], s=subset_bubble_sizes, color=color, alpha=0.6, edgecolors="w", linewidth=0.5)
    ax.scatter([], [], s=150, color=color, label=continent, alpha=0.6, edgecolors="w", linewidth=0.5)

ax.text(0,1.05, 'a. Forest Edge Dynamics (Country level)', transform=ax.transAxes, fontsize = 7,fontweight='bold')
ax.tick_params(axis='both',which='major',labelsize=6,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,
                bottom=True,left=True,top=False,right=False)

ax.set_xlabel('Edge Change (2020-2000)/2000',fontsize=7,labelpad = 1, fontweight='bold')
ax.set_ylabel('Area Change (2020-2000)/2000',fontsize=7,labelpad = 1, fontweight='bold')

legend = ax.legend()
handles = legend.legendHandles
labels = [text.get_text() for text in legend.get_texts()]
original_colors = [handle.get_facecolor() for handle in handles]
new_handles = [Line2D([0], [0], marker='o', label=label, color=color, markersize=4, linestyle='None') for label, color in zip(labels, original_colors)]
legend.remove()
ax.legend(handles=new_handles, labels=labels, title='Continent', bbox_to_anchor=(0, 0.97),
           title_fontsize=6, loc='upper left', fontsize=5, facecolor='none', edgecolor='none')

ax.axvline(0, color='black', linestyle='-', linewidth=1)  # x=0 line
ax.axhline(0, color='black', linestyle='-', linewidth=1)  # y=0 line

axes = inset_axes(ax, width="30%", height="30%", loc='lower right', bbox_to_anchor=(-0.02, 0.6, 1, 1),bbox_transform=ax.transAxes)
for continent, color in colors.items():
    outliers = df[(abs(df['Edge Change']) > edge_change_threshold)
        | (abs(df['Area Change']) > area_change_threshold)]
    subset = outliers[outliers['continent'] == continent]
    subset_bubble_sizes = bubble_sizes[subset.index]
    axes.scatter(subset['Edge Change'], subset['Area Change'], s=subset_bubble_sizes, color=color, alpha=0.6, edgecolors="w", linewidth=0.5)
    
axes.set_title('Outliers',fontsize = 6)
axes.axvline(0, color='black', linestyle='-', linewidth=1) 
axes.axhline(0, color='black', linestyle='-', linewidth=1)
axes.tick_params(axis='both',which='major',labelsize=8,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,
                bottom=True,left=True,top=False,right=False)

#**********
ax.text(1.1,1.05, 'b. Forest Edge Dynamics in Hotspot Countries', transform=ax.transAxes, fontsize = 7,fontweight='bold')

axes = [ax1, ax2, ax3, ax4, ax5, ax6]
for i, axe in enumerate(axes):
    temp = final_df[final_df["Country"]==selected_countries[i]]
    
    sns.barplot(x='time_period', y= "length", hue = 'len_type', ax = axe, data=temp, palette=[bivariate_colors[n-1][n-1],bivariate_colors[0][n-1], bivariate_colors[n-1][0], bivariate_colors[0][0]],
                saturation=0.7, errcolor='k',errwidth = 0.7,capsize = 0.07,edgecolor="k",linewidth = 0.4)
    
    axe.legend(loc = 'lower right',fontsize=6,facecolor= 'none',edgecolor = 'none',bbox_to_anchor=(0.4, -0.65), ncol = 2,columnspacing = 0.4)
    
    axe.ticklabel_format(style='scientific', scilimits=(0, 0), axis='y')
    axe.yaxis.offsetText.set_visible(False)
    for tick in axe.get_xticklabels():
            tick.set_rotation(20)
    axe.set_xlabel('')
    expo = [6,6,6,5,6,5]
    axe.set_ylabel(f"Forest edge length (km) $\\times 10^{{{expo[i]}}}$", labelpad = 0.1, fontsize = 5)
    
    axe.spines[['top', 'right']].set_visible(False)
    axe.tick_params(axis='both',which='major',labelsize=5,direction='in',labeltop=False,
                labelbottom=True,pad=1,bottom=True, left=True,top=False,right=False)
    axe.text(0.03,0.97, f'(b.{i+1}) {selected_countries[i]}', transform=axe.transAxes, fontsize = 6)
    if i !=5:
        axe.legend_.remove()
    else:
        axe.text(-1, -0.3, "Year", transform=axe.transAxes, fontsize = 7,fontweight='bold')
# Fully vector line art -> PDF (submission) + 300-dpi PNG (preview); text stays editable.
out = '/scratch/fji7/Forest_edge_mapping_2024_11_14/2_exported_figures/Figure 2_Forest edge dynamics statistics'
plt.savefig(out + '.pdf', dpi=600, bbox_inches='tight')
plt.savefig(out + '.png', dpi=600, bbox_inches='tight')