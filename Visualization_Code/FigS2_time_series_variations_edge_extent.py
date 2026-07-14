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


file_name = "/scratch/fji7/Forest_edge_mapping_2024_11_14/1_data/Edge_climate_zone_continent_stats.csv"

df = pd.read_csv(file_name)

merged_row = df[df["Climate Zone"].isin(["Australia", "Oceania"])].sum(numeric_only=True)
merged_row["Climate Zone"] = "Oceania"
df = df[~df["Climate Zone"].isin(["Australia", "Oceania"])]
df = df.append(merged_row, ignore_index=True)

df.rename(columns={"Climate Zone": "Type"}, inplace=True)

df1 = df.iloc[0:6,0:11]
df2 = df.iloc[7:,0:11]

years = [2000, 2005, 2010, 2015, 2020]
start_var = True
for ty in df1["Type"].unique():
    temp = df1[df1["Type"] == ty]
    edge = pd.DataFrame(temp.iloc[:,1:6].values)
    edge = edge.T
    edge.columns = ["values"]
    edge["years"] = years
    edge["type"] = ty
    
    area = pd.DataFrame(temp.iloc[:,6:].values)
    area = area.T
    area.columns = ["values"]
    area["years"] = years
    area["type"] = ty
    if start_var:
        df1_edge = edge
        df1_area = area
        start_var = False
    else:
        df1_edge = pd.concat([df1_edge, edge], axis = 0)
        df1_area = pd.concat([df1_area, area], axis = 0)
df1_edge.reset_index(drop = True, inplace = True)
df1_area.reset_index(drop = True, inplace = True)

#****************************************************************
start_var = True
for ty in df2["Type"].unique():
    temp = df2[df2["Type"] == ty]
    edge = pd.DataFrame(temp.iloc[:,1:6].values)
    edge = edge.T
    edge.columns = ["values"]
    edge["years"] = years
    edge["type"] = ty
    
    area = pd.DataFrame(temp.iloc[:,6:].values)
    area = area.T
    area.columns = ["values"]
    area["years"] = years
    area["type"] = ty
    if start_var:
        df2_edge = edge
        df2_area = area
        start_var = False
    else:
        df2_edge = pd.concat([df2_edge, edge], axis = 0)
        df2_area = pd.concat([df2_area, area], axis = 0)
df2_edge.reset_index(drop = True, inplace = True)
df2_area.reset_index(drop = True, inplace = True)

df1_edge["data"] = "edge"
df1_area["data"] = "area"

df2_edge["data"] = "edge"
df2_area["data"] = "area"

df = pd.concat([df1_edge, df1_area, df2_edge, df2_area], axis = 0)
df.reset_index(drop = True, inplace = True)


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
fig = plt.figure(figsize = (200*mm, 185*mm))   # oversized canvas; tight-crop lands ~180 mm wide
gs = gridspec.GridSpec(32, 25)
plt.subplots_adjust(hspace =0.6,wspace =0.5)

ax1 = plt.subplot(gs[0:5, 0:11])
ax2 = plt.subplot(gs[0:5, 14:25])
ax3 = plt.subplot(gs[5:10, 0:11])
ax4 = plt.subplot(gs[5:10, 14:25])
ax5 = plt.subplot(gs[10:15, 0:11])
ax6 = plt.subplot(gs[10:15, 14:25])
ax7 = plt.subplot(gs[17:22, 0:11])
ax8 = plt.subplot(gs[17:22, 14:25])
ax9 = plt.subplot(gs[22:27, 0:11])
ax10 = plt.subplot(gs[22:27, 14:25])
ax11 = plt.subplot(gs[27:32, 0:11])
ax12 = plt.subplot(gs[27:32, 14:25])


axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]
years = ["2000", "2005", "2010", "2015", "2020"]

for idx, ty in enumerate(df["type"].unique()):
    ax = axes[idx]
    axe = ax.twinx()
    
    data = df[df["type"] == ty]
    edge = data[data["data"]=="edge"]
    area = data[data["data"]=="area"]
    
    ax.plot(years, area["values"], color = "royalblue", marker = 'o',markersize=4,linewidth=1.5, label= r'Forest areas ($km^2$)')
    axe.plot(years, edge["values"], color = "orangered", marker = 's',markersize=3,linewidth=1.5,ls = "-.", label='Forest edge length (km)')
    
    if idx != 11:
        ax.set_ylim(0.98*area["values"].min(), 1.02*area["values"].max())
        axe.set_ylim(0.98*edge["values"].min(), 1.02*edge["values"].max())
    else:
        ax.set_ylim(0.995*area["values"].min(), 1.005*area["values"].max())
        axe.set_ylim(0.995*edge["values"].min(), 1.005*edge["values"].max())
    
    ax.tick_params(axis='y', labelsize=6, labelcolor='royalblue')
    ax.ticklabel_format(style='scientific', scilimits=(0, 0), axis='y')
    plt.draw()
    offset_text = ax.yaxis.offsetText.get_text()
    _expa = int(offset_text[2:]) if len(offset_text) > 2 else 0
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, p, e=_expa: f"{v/10**e:.2f}"))  # uniform 2-decimal ticks
    ax.set_xlabel('Year',fontsize = 7,labelpad = 1)
    ax.set_ylabel(f"Forest areas ($km^2$) $\\times 10^{{{offset_text[2:]}}}$", color='royalblue', labelpad = 0.1, fontsize = 5)
    ax.yaxis.offsetText.set_visible(False)
    
    axe.tick_params(axis='y', labelsize=6, labelcolor='orangered')
    axe.ticklabel_format(style='scientific', scilimits=(0, 0), axis='y')
    plt.draw()
    offset_text = axe.yaxis.offsetText.get_text()
    _expe = int(offset_text[2:]) if len(offset_text) > 2 else 0
    axe.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, p, e=_expe: f"{v/10**e:.2f}"))  # uniform 2-decimal ticks
    axe.set_ylabel(f"Forest edge \nlength ($km$) $\\times 10^{{{offset_text[2:]}}}$", color='orangered', labelpad = 3, fontsize = 5)
    axe.yaxis.offsetText.set_visible(False)
    
    legend1 = ax.legend(loc = 'lower left',fontsize=6, facecolor= 'none',edgecolor = 'none',bbox_to_anchor=(0, 0.15))
    legend2 = axe.legend(loc = 'lower left',fontsize=6, facecolor= 'none',edgecolor = 'none',bbox_to_anchor=(0, 0))
    for text in legend1.get_texts():
        text.set_color("royalblue")
    for text in legend2.get_texts():
        text.set_color("orangered")
        
    if idx != 0:
        ax.legend_.remove()
        axe.legend_.remove()
    
    if (idx!=4)&(idx!=5)&(idx!=10)&(idx!=11):
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    if idx == 0:
        ax.text(0,1.05, 'a. Forest Areas and Length Variations Over Time in Different Climate Zones', transform=ax.transAxes, fontsize = 9,fontweight='bold')
    if idx == 6:
        ax.text(0,1.07, 'b. Forest Areas and Length Variations Over Time in Different Continents', transform=ax.transAxes, fontsize = 9,fontweight='bold') 
        
    ax.text(0.01,0.91, f"(a.{idx+1}) {ty}", transform=ax.transAxes, fontsize = 6) if idx in [0,1,2,3,4,5] else ax.text(0.01,0.91, f"(b.{idx-5}) {ty}", transform=ax.transAxes, fontsize = 6)
out = '/scratch/fji7/Forest_edge_mapping_2024_11_14/2_exported_figures/Figure S2_Time series variations edge extent'
plt.savefig(out + '.pdf', dpi=600, bbox_inches='tight')
plt.savefig(out + '.png', dpi=600, bbox_inches='tight')
