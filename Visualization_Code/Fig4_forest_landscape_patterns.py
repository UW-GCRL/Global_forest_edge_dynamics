import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.transforms as mtransforms
from scipy import stats
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from shapely.geometry import MultiPolygon


def get_color(deviation, max_deviation):
    color_intensity = np.abs(deviation) / max_deviation
    if deviation > 0:
        return (1, 1-color_intensity, 1-color_intensity)  # shades of red
    elif deviation < 0:
        return (1-color_intensity, 1-color_intensity, 1)  # shades of blue
    else:
        return 'grey'
    
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

#***************************************************************************************************************************************
file_path = '/scratch/fji7/Forest_edge_mapping_2024_11_14/1_data/processed_country_data_with_area.csv'
country_data = pd.read_csv(file_path)

china_row = country_data[country_data['country'] == 'China']
taiwan_row = country_data[country_data['country'] == 'Taiwan']

if not china_row.empty and not taiwan_row.empty:
    columns_to_sum = country_data.columns.difference(['country'])
    for column in columns_to_sum:
        country_data.loc[country_data['country'] == 'China', column] += taiwan_row[column].values[0]
    country_data = country_data[country_data['country'] != 'Taiwan']

    
log_forest_area = np.log(country_data['Total Forest Area 2000'])
log_forest_edge = np.log(country_data['forest edge 2000'])

slope, intercept, r_value, p_value, std_err = stats.linregress(log_forest_area, log_forest_edge)

def log_log_regression_model(log_x):
    return intercept + slope * log_x

log_estimated_forest_edge = log_log_regression_model(log_forest_area)
estimated_forest_edge = np.exp(log_estimated_forest_edge)

country_data['log_residuals2000'] = log_forest_edge - log_estimated_forest_edge
country_data['log_fragmentation_rank2000'] = country_data['log_residuals2000'].abs().rank(ascending=False)


log_forest_edge_2020 = np.log(country_data['forest edge 2020'])
log_forest_area_2020 = np.log(country_data['Total Forest Area 2020'])

log_estimated_forest_edge_2020 = log_log_regression_model(log_forest_area_2020)
estimated_forest_edge_2020 = np.exp(log_estimated_forest_edge_2020)

country_data['log_residuals2020'] = log_forest_edge_2020 - log_estimated_forest_edge_2020
country_data['log_fragmentation_rank2020'] = country_data['log_residuals2020'].abs().rank(ascending=False)

#***************************************************************************************************************************************

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
# so this lands at ~180 x 216 mm (2-column width) with text at 5-7 pt.
fig = plt.figure(figsize=(306*mm, 245*mm))
plt.subplots_adjust(hspace =0.15)

ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson(central_longitude=0.0))
base_map(ax1)

grd = ax1.gridlines(draw_labels=True, xlocs=range(-180, 181, 90), ylocs=range(-60, 61, 30), color='gray',linestyle='--', linewidth=0.5, zorder=7)
# cartopy 0.18 mis-places longitude labels on the curved Robinson boundary; blank them and
# add plain horizontal ones on the bottom edge. Match latitude labels to the 6 pt longitude.
grd.top_labels = False
grd.ylabel_style = {'size': 6}
grd.xformatter = mtick.FuncFormatter(lambda v, pos: '')
_bx0, _bx1, _blat, _btop = ax1.get_extent(ccrs.PlateCarree())
_botlab = mtransforms.offset_copy(ax1.transData, fig=fig, y=-4, units='points')
for _lon, _lab in [(-180, '180°'), (-90, '90°W'), (0, '0°'), (90, '90°E'), (180, '180°')]:
    _x, _y = ax1.projection.transform_point(_lon, _blat, ccrs.PlateCarree())
    ax1.text(_x, _y, _lab, transform=_botlab, ha='center', va='top', fontsize=6, color='k', zorder=8)

ax1.text(-0.02,1.02, 'a. Disparities in Forest Edge Length Relative to Area at Country Scale in 2000', transform=ax1.transAxes, fontsize = 9,fontweight='bold')
countries_shp = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
max_deviation = max(abs(country_data['log_residuals2000']))
        
for country in shpreader.Reader(countries_shp).records():
    country_name = country.attributes['SOVEREIGNT']
    if country_name in country_data['country'].values:
        deviation = country_data[country_data['country'] == country_name]['log_residuals2000'].values[0]
        color = get_color(deviation, max_deviation)
        geometry = country.geometry
        if isinstance(geometry, MultiPolygon):
            geometries = geometry.geoms
        else:
            geometries = [geometry]  

        ax1.add_geometries(geometries, ccrs.PlateCarree(), facecolor=color)

    if country_name == "Taiwan":
        deviation = country_data[country_data['country'] == "China"]['log_residuals2000'].values[0]
        color = get_color(deviation, max_deviation)

        geometry = country.geometry
        if isinstance(geometry, MultiPolygon):
            geometries = geometry.geoms
        else:
            geometries = [geometry]

        ax1.add_geometries(geometries, ccrs.PlateCarree(), facecolor=color)        
              
ax1.add_feature(cf.BORDERS, linestyle='-', alpha=1,edgecolor='black', lw=0.5)
ax1.add_feature(cf.COASTLINE, lw=0.5,alpha=1,edgecolor='black')

norm1 = mpl.colors.Normalize(vmin=-max_deviation, vmax=max_deviation)
cmap1 = mpl.cm.RdBu_r
scalar_map1 = mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1)
cbar1 = plt.colorbar(scalar_map1, ax=ax1, orientation='horizontal', fraction=0.04, pad=0.05)
cbar1.outline.set_visible(False)
cbar1.ax.tick_params(labelsize=6,pad = 0.1,length=1)
cbar1.set_label('Forest Edge Length Residuals at Log Scale (log(km))',fontsize = 7,labelpad=2)

ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.Robinson(central_longitude=0.0))
ax2.set_extent([-179.999, 179.999, -90, 90])
base_map(ax2)

grd = ax2.gridlines(draw_labels=True, xlocs=range(-180, 181, 90), ylocs=range(-60, 61, 30), color='gray',linestyle='--', linewidth=0.5, zorder=7)
grd.top_labels = False
grd.ylabel_style = {'size': 6}
grd.xformatter = mtick.FuncFormatter(lambda v, pos: '')
_bx0, _bx1, _blat, _btop = ax2.get_extent(ccrs.PlateCarree())
_botlab = mtransforms.offset_copy(ax2.transData, fig=fig, y=-4, units='points')
for _lon, _lab in [(-180, '180°'), (-90, '90°W'), (0, '0°'), (90, '90°E'), (180, '180°')]:
    _x, _y = ax2.projection.transform_point(_lon, _blat, ccrs.PlateCarree())
    ax2.text(_x, _y, _lab, transform=_botlab, ha='center', va='top', fontsize=6, color='k', zorder=8)

ax2.text(-0.02,1.02, 'b. Dynamics of Disparities in Forest Edge Length Relative to Area at Country Scale', transform=ax2.transAxes, fontsize = 9,fontweight='bold')

max_deviation_diff = max(abs(country_data['log_residuals2020'] - country_data['log_residuals2000']))
        
for country in shpreader.Reader(countries_shp).records():
    country_name = country.attributes['SOVEREIGNT']
    if country_name in country_data['country'].values:
        deviation = country_data[country_data['country'] == country_name]['log_residuals2020'].values[0] - country_data[country_data['country'] == country_name]['log_residuals2000'].values[0]
        color = get_color(deviation, max_deviation_diff)
        geometry = country.geometry
        if isinstance(geometry, MultiPolygon):
            geometries = geometry.geoms
        else:
            geometries = [geometry]  

        ax2.add_geometries(geometries, ccrs.PlateCarree(), facecolor=color)

    if country_name == "Taiwan":
        deviation = country_data[country_data['country'] == "China"]['log_residuals2020'].values[0] - country_data[country_data['country'] == "China"]['log_residuals2000'].values[0]
        color = get_color(deviation, max_deviation_diff)

        geometry = country.geometry
        if isinstance(geometry, MultiPolygon):
            geometries = geometry.geoms
        else:
            geometries = [geometry]

        ax2.add_geometries(geometries, ccrs.PlateCarree(), facecolor=color)  
#"""
ax2.add_feature(cf.BORDERS, linestyle='-', alpha=1,edgecolor='black', lw=0.5)
ax2.add_feature(cf.COASTLINE, lw=0.5,alpha=1,edgecolor='black')

norm2 = mpl.colors.Normalize(vmin=-max_deviation_diff, vmax=max_deviation_diff)
cmap2 = mpl.cm.RdBu_r
scalar_map2 = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2)
cbar2 = plt.colorbar(scalar_map2, ax=ax2, orientation='horizontal', fraction=0.04, pad=0.05)
cbar2.outline.set_visible(False)
cbar2.ax.tick_params(labelsize=6,pad = 0.1,length=1)
cbar2.set_label('\u0394 Forest Edge Length Residuals at Log Scale (log(km))',fontsize = 7,labelpad=2)
# Fully vector (country fills are vector polygons) -> PDF (submission) + 300-dpi PNG preview.
out = '/scratch/fji7/Forest_edge_mapping_2024_11_14/2_exported_figures/Figure 4_Forest landscape patterns'
plt.savefig(out + '.pdf', dpi=600, bbox_inches='tight')
plt.savefig(out + '.png', dpi=600, bbox_inches='tight')