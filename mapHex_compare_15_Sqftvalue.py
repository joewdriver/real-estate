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
import operator
from sklearn import preprocessing


tic = time.clock()

data2008 = pd.read_csv('./new_data/data2018.csv')
data = pd.read_csv('./new_data/data2015.csv')
#print data2008
#print data

# Get specific columns that we will be using and transfer them into the appropriate type
# For training data
lat = data["latitude"] 
lon = data["longitude"]
year = data["year"]
bdrms = data["bedrooms"]
fbath = data["full_bth"]
hbath = data["half_bth"]
sf = data["square_foot"]
res = data["res"]
condo = data["condo"]
built = data["yr_built"]
bldg = data["bldg_price"]
land = data["land_price"]
data['total'] =  np.add(bldg, land)
#per_sq = np.divide(total, sf)


lon.astype('float')
year.astype('float')
bdrms.astype('float')
fbath.astype('float')
hbath.astype('float')
sf.astype('float')
res.astype('float')
condo.astype('float')
built.astype('float')
bldg.astype('float')
land.astype('float')
#per_sq.astype('float')

# Get specific columns that we will be using and transfer them into the appropriate type
# For testing data
lat_2008 = data2008["latitude"]
lon_2008 = data2008["longitude"]
year_2008 = data2008["year"]
bdrms_2008 = data2008["bedrooms"]
fbath_2008 = data2008["full_bth"]
hbath_2008 = data2008["half_bth"]
sf_2008 = data2008["square_foot"]
res_2008 = data2008["res"]
condo_2008 = data2008["condo"]
built_2008 = data2008["yr_built"]
bldg_2008 = data2008["bldg_price"]
land_2008 = data2008["land_price"]
data2008['total'] = np.add(bldg_2008, land_2008)
#per_sq_2008 = np.divide(total_2008, sf_2008)

lat_2008.astype('float')
lon_2008.astype('float')
year_2008.astype('float')
bdrms_2008.astype('float')
fbath_2008.astype('float')
hbath_2008.astype('float')
sf_2008.astype('float')
res_2008.astype('float')
condo_2008.astype('float')
built_2008.astype('float')
bldg_2008.astype('float')
land_2008.astype('float')
#per_sq_2008.astype('float')

# Concatenate all values above
#df2015 = pd.concat([lat,lon,year,bdrms,fbath,hbath,sf,res,condo,built, total], axis = 1)
df= pd.concat([lat,lon, built, sf,], axis = 1)
df['bldg_price2015'] = data['total']

#print df
#getting rid of all of the zero values in square foot, so we don't divide by zero
df = df[df['square_foot'] !=0.0]

#For the square footage
df['AV_TOTAL'] = df['bldg_price2015']/df['square_foot']

#print len(df['AV_TOTAL'])

#getting rid of outliers
over = df['AV_TOTAL'].quantile(0.95)
under = df['AV_TOTAL'].quantile(0.05)


#print over,under
df = df[df['AV_TOTAL'] <= over] #change back to over
df = df[df['AV_TOTAL'] >= under] #change back to under
#df['AV_ORIG'] = df['AV_TOTAL']
#df['AV_TOTAL'][df['AV_ORIG'] > 70.0]= 70.0

#print df['AV_ORIG'], df['AV_TOTAL']

#storing the quartiles of non-normalized data for colorbar label
qt10 = df['AV_TOTAL'].quantile(0.10)
qt20 = df['AV_TOTAL'].quantile(0.20)
qt30 = df['AV_TOTAL'].quantile(0.30)
qt40 = df['AV_TOTAL'].quantile(0.40)
qt50 = df['AV_TOTAL'].quantile(0.50)
qt60 = df['AV_TOTAL'].quantile(0.60)
qt70 = df['AV_TOTAL'].quantile(0.70)
qt80 = df['AV_TOTAL'].quantile(0.80)
qt90 = df['AV_TOTAL'].quantile(0.90)


print qt10
print qt20
print qt30
print qt40
print qt50
print qt60
print qt70
print qt80
print qt90

#normalize the data

max_value = df['AV_TOTAL'].max()
min_value = df['AV_TOTAL'].min()
#print max_value,min_value
df['AV_TOTAL'] = (df['AV_TOTAL'] - min_value) / (max_value - min_value)


#print df['AV_TOTAL'],df['AV_ORIG']
##scaling the values to make it easier to plot
#df['AV_TOTAL'] = (df['AV_TOTAL'].astype(int))



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
#print df_map['hood_count']

# # # # 
# We'll only use a handful of distinct colors for our choropleth. So pick where
# you want your cutoffs to occur. Leave zero and ~infinity alone.
breaks = [0.] + [qt10, qt20, qt30, qt40, qt50, qt60, qt70, qt80, qt90] + [1e20]
def self_categorize(entry, breaks):
    for i in range(len(breaks)-1):
        if entry > breaks[i] and entry <= breaks[i+1]:
            return i
    return -1
df_map['jenks_bins'] = df_map.hood_count.apply(self_categorize, args=(breaks,))


# # # Or, you could always use Natural_Breaks to calculate your breaks for you:
# # from pysal.esda.mapclassify import Natural_Breaks
# # breaks = Natural_Breaks(df_map[df_map['hood_count'] > 0].hood_count, initial=300, k=6)
# # df_map['jenks_bins'] = -1 #default value if no data exists for this bin
# # df_map['jenks_bins'][df_map.hood_count > 0] = breaks.yb

qt10 = int(qt10)
qt20 = int(qt20)
qt30= int(qt30)
qt40= int(qt40)
qt50= int(qt50)
qt60= int(qt60)
qt70= int(qt70)
qt80= int(qt80)
qt90= int(qt90)

jenks_labels = ['Little Change']+["> $%d"%(perc) for perc in breaks[:-1]]
#jenks_labels = ["Lowest Increase"] + ['> $' str(qt10), '> $' str(qt10), '> $' str(qt10), '> $' str(qt10), '> $' str(qt10), '> $' str(qt10),'> $' str(qt10), '> $' str(qt10),'> $' str(qt10)] + ['Largest Increase']
#jenks_labels = ["> 0"]+["> %d"%(perc) for perc in breaks.bins[:-1]]

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
cbar = custom_colorbar(cmap, ncolors=len(jenks_labels)+1, labels=jenks_labels, shrink=0.5)
#cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Change in value/Sq. Ft.', rotation=270)


##fig.suptitle("My location density in Seattle", fontdict={'size':24, 'fontweight':'bold'}, y=0.92)
ax.set_title("Boston Housing Predicted Change in Square Ft. Value Between 2018 and 2021", fontsize=14, y=0.98)

plt.savefig('./images/PerSq_norm_sqftvalue15.png', dpi=300, frameon=False, bbox_inches='tight', pad_inches=0.5, facecolor='#DEDEDE')
plt.show()


toc = time.clock()

comptime = toc-tic
print comptime
