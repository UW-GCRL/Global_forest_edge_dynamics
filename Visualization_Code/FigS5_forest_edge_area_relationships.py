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


def rsquared(x, y): 
    """Return the metriscs coefficient of determination (R2)
    Parameters:
    -----------
    x (numpy array or list): Predicted variables
    y (numpy array or list): Observed variables
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
    a = r_value**2
    return a

#*************************
file_name = "/scratch/fji7/Forest_edge_mapping_2024_11_14/1_data/Country_Forest_Area_Edge.csv"
df = pd.read_csv(file_name)
df.dropna(inplace = True)


x1, x2, x3, x4, x5 = df["Total Forest Edge Length 2000 (KM)"],df["Total Forest Edge Length 2005 (KM)"],df["Total Forest Edge Length 2010 (KM)"],df["Total Forest Edge Length 2015 (KM)"],df["Total Forest Edge Length 2020 (KM)"]
y1, y2, y3, y4, y5 = df["Total Forest Area 2000 (KM2)"],df["Total Forest Area 2005 (KM2)"],df["Total Forest Area 2010 (KM2)"],df["Total Forest Area 2015 (KM2)"],df["Total Forest Area 2020 (KM2)"]

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
fig,ax = plt.subplots(figsize = (110*mm, 88*mm))   # oversized; tight-crop lands ~88 mm (1-column)
plt.subplots_adjust(hspace =0.2,wspace =0.2)

x = [x1, x2, x3, x4, x5]
y = [y1, y2, y3, y4, y5]
colors = ["orangered", "#35a153","#303cf9", "#979797", "dodgerblue"]
markers = ["P", "X", "^", "v", "o"]
years = ["2000", "2005", "2010", "2015", "2020"]

a_total = []
b_total = []
for idx in range(5):
    a,b = x[idx], y[idx]
    a_total.extend(a)
    b_total.extend(b)
    
    ax.scatter(a,b,color=colors[idx], label=years[idx], alpha=0.7, edgecolors='w', linewidth=0.5, marker = markers[idx],s = 50)
    

final = pd.DataFrame([a_total,b_total]).T
final.columns = ["edge","area"]
final = final[(final['edge'] > 0) & (final['area'] > 0)]

coeffs = np.polyfit(np.log(final['edge']), np.log(final['area']), 1)
print(coeffs)
ax.plot(final['edge'], np.exp(coeffs[1]) * final['edge'] ** coeffs[0], color='k', linestyle='-', linewidth=2)

# R2 = rsquared(final['edge'], final['area'])
# _, p_value = stats.ttest_ind(final['edge'], final['area'])
R2 = rsquared(df["Total Forest Edge Length 2000 (KM)"], df["Total Forest Area 2000 (KM2)"])
_, p_value = stats.ttest_ind(df["Total Forest Edge Length 2000 (KM)"], df["Total Forest Area 2000 (KM2)"])

ax.text(0.02, 0.93, f"$\\mathit{{R}}^2$ = {round(R2,3)}", transform=ax.transAxes, color='k', fontsize=7)
# ax.text(0.02, 0.86, f"$p$ = {round(p_value,5)}", transform=ax.transAxes, color='k', fontsize=7)

p_str = "{:.2e}".format(p_value)  # 保留2位小数
mantissa, exponent = p_str.split("e")
exponent = int(exponent)
text_str = f"$\\mathit{{p}} = {mantissa}\\times10^{{{exponent}}}$"
ax.text(0.02, 0.86, text_str, transform=ax.transAxes, color='k', fontsize=8)
ax.text(0.02, 0.79, f"$\\mathit{{N}}$ = {len(df)}", transform=ax.transAxes, color='k', fontsize=8)

ax.set_xlabel('Edge ($km$)',fontsize=7,labelpad = 1)
ax.set_ylabel('Area ($km^2$)',fontsize=7,labelpad = 1)
ax.legend(title='Year',title_fontsize=7, scatterpoints=1, loc = 'lower right',fontsize=6,facecolor= 'none',edgecolor = 'none')
ax.tick_params(axis='both',which='major',labelsize=6,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,
            bottom=True,left=True,top=False,right=False)
ax.set_xscale('log')
ax.set_yscale('log')

out = '/scratch/fji7/Forest_edge_mapping_2024_11_14/2_exported_figures/Figure S5_forest edge_area relationships'
plt.savefig(out + '.pdf', dpi=600, bbox_inches='tight')
plt.savefig(out + '.png', dpi=600, bbox_inches='tight')