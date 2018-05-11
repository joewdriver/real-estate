#first try
#from https://sensitivecities.com/so-youd-like-to-make-a-map-using-python-EN.html


from lxml import etree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from pysal.esda.mapclassify import Natural_Breaks as nb
from descartes import PolygonPatch
import fiona
from itertools import chain
import time

tic = time.clock()


data = pd.read_csv('property-assessment-fy2015.csv')


# TODO: Need to filter all data

# Values to predict on
locations = data['Location']

# Prices
bldg_price = data[['AV_BLDG']].copy()
land_price = data[['AV_LAND']].copy()
total_price = data[['AV_TOTAL']].copy()

# TODO:Add LU

lat = []
lon = []
prev = 0
for i in range(len(locations)):

    # TODO: TEMP FIX, no lat/lon.
    if type(locations[i]) == type(0.0) or locations[i] == "0":
        a,b = locations[prev].split("|")
        lat.append(float(a[1:]))
        lon.append(float(b[:-1]))

    else:
        a,b = locations[i].split("|")
        lat.append(float(a[1:]))
        lon.append(float(b[:-1]))
        prev = i




lat = pd.Series(lat,name="latitude")
lon = pd.Series(lon,name="longitude")

df = pd.concat([bldg_price,land_price,total_price,lat,lon], axis = 1)

#Deleting all of the 0 values for now
df = df[df.latitude !=0.0]
df = df[df['AV_TOTAL'] !=0.0]
df = df[~df.isin(['NaN']).any(axis=1)]
#scaling the values to make it easier to plot

df['AV_TOTAL'] = (df['AV_TOTAL'].astype(int))

#print df

shp = fiona.open('Boston_Neighborhoods.shp')

bds = shp.bounds
shp.close()
extra = 0.01
ll = (bds[0], bds[1])
ur = (bds[2], bds[3])
coords = list(chain(ll, ur))
w, h = coords[2] - coords[0], coords[3] - coords[1]

m = Basemap(
    projection='tmerc',
    lon_0=-71.04,
    lat_0=42.35,
    ellps = 'WGS84',
    llcrnrlon=coords[0] - extra * w,
    llcrnrlat=coords[1] - extra + 0.01 * h,
    urcrnrlon=coords[2] + extra * w,
    urcrnrlat=coords[3] + extra + 0.01 * h,
    lat_ts=0,
    resolution='i',
    suppress_ticks=True)
m.readshapefile(
    'Boston_Neighborhoods',
    'boston',
    drawbounds=False,
    color='none',
    zorder=2)



# set up a map dataframe
df_map = pd.DataFrame({
    'poly': [Polygon(xy) for xy in m.boston],
    'hoodname': [names['Neighborho'] for names in m.boston_info]})
df_map['SqMiles'] = df_map['poly'].map(lambda x:x.area)



# Convert our latitude and longitude into Basemap cartesian map coordinates
mapped_points = [Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(df['longitude'], 
            df['latitude'])]
all_points = MultiPoint(mapped_points)
# Use prep to optimize polygons for faster computation


def num_of_contained_points(apolygon, all_points):
    return int(len(filter(prep(apolygon).contains, all_points)))

df_map['hood_count'] = df_map['poly'].apply(num_of_contained_points, args=(all_points,))


# We'll only use a handful of distinct colors for our choropleth. So pick where
# you want your cutoffs to occur. Leave zero and ~infinity alone.
breaks = [0.] + [25., 50., 75., 100., 125., 150., 200.,250.,300.,350.,400.] + [1e20]
def self_categorize(entry, breaks):
    for i in range(len(breaks)-1):
        if entry > breaks[i] and entry <= breaks[i+1]:
            return i
    return -1
df_map['jenks_bins'] = df_map.hood_count.apply(self_categorize, args=(breaks,))


# Or, you could always use Natural_Breaks to calculate your breaks for you:
from pysal.esda.mapclassify import Natural_Breaks
breaks = Natural_Breaks(df_map[df_map['hood_count'] > 0].hood_count, initial=300, k=6)
df_map['jenks_bins'] = -1 #default value if no data exists for this bin
df_map['jenks_bins'][df_map.hood_count > 0] = breaks.yb

# # #jenks_labels = [0.] + [2.5, 5.0, 7.5, 10., 12.5, 15., 20.,25.,30.,35.,40.] + ['Above']
jenks_labels = ["> 0"]+["> %d"%(perc) for perc in breaks.bins[:-1]]

def custom_colorbar(cmap, ncolors, labels, **kwargs):    
    """Create a custom, discretized colorbar with correctly formatted/aligned labels.
    
    cmap: the matplotlib colormap object you plan on using for your graph
    ncolors: (int) the number of discrete colors available
    labels: the list of labels for the colorbar. Should be the same length as ncolors.
    """
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable
        
    norm = BoundaryNorm(range(0, ncolors), cmap.N)
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors+1)+0.5)
    colorbar.set_ticklabels(range(0, ncolors))
    colorbar.set_ticklabels(labels)
    return colorbar



cmap = plt.get_cmap('cool')


"""PLOT A HEXBIN MAP OF LOCATION
"""
figwidth = 14
fig = plt.figure(figsize=(figwidth, figwidth*h/w))
ax = fig.add_subplot(111, frame_on=False)

# draw neighborhood patches from polygons
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
    x, fc='#555555', ec='#555555', lw=1, alpha=1, zorder=0))
# plot neighborhoods by adding the PatchCollection to the axes instance
ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))

# the mincnt argument only shows cells with a value >= 1
# The number of hexbins you want in the x-direction
numhexbins = 100

hx = m.hexbin(
    np.array([geom.x for geom in all_points]),
    np.array([geom.y for geom in all_points]),
    C=df['AV_TOTAL'],
    gridsize=(numhexbins, int(numhexbins*h/w)), #critical to get regular hexagon, must stretch to map dimensions
    bins='log', mincnt=1, edgecolor='none', alpha=1.,
    cmap=plt.get_cmap('cool'))


# Draw the patches again, but this time just their borders (to achieve borders over the hexbins)
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
    x, fc='none', ec='#FFFF99', lw=1, alpha=1, zorder=1))
ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))

# Draw a map scale
m.drawmapscale(coords[0] + 0.05, coords[1] - 0.01,
    coords[0], coords[1], 4.,
    units='mi', barstyle='fancy', labelstyle='simple',
    fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
    zorder=5)

#ncolors+1 because we're using a "zero-th" color
#cbar = custom_colorbar(cmap, ncolors=len(jenks_labels)+1, labels=jenks_labels, shrink=0.5)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)

##fig.suptitle("My location density in Seattle", fontdict={'size':24, 'fontweight':'bold'}, y=0.92)
##ax.set_title("Using location data collected from my Android phone via Google Takeout", fontsize=14, y=0.98)
##ax.text(1.0, 0.03, "Collected from 2012-2014 on Android 4.2-4.4\nGeographic data provided by data.seattle.gov", 
##        ha='right', color='#555555', style='italic', transform=ax.transAxes)
##ax.text(1.0, 0.01, "BeneathData.com", color='#555555', fontsize=16, ha='right', transform=ax.transAxes)
plt.savefig('data\hexbin3_norm.png', dpi=600, frameon=False, bbox_inches='tight', pad_inches=0.5, facecolor='#DEDEDE')
plt.show()


toc = time.clock()

comptime = toc-tic
print comptime