import matplotlib.pyplot as plt
from matplotlib import cbook
import time
import visualization_on_map

left = 2+7/24
right = 6+11/18
bottom = 49+1/3
up = 51+7/12

nb_line = 0
start = time.time()

datafile = cbook.get_sample_data('C:\\Users\\P62116\\Documents\\Code\\map_gris.png')
img = plt.imread(datafile)

plt.figure()

search_file = open("searchCriterion.csv")
searches_coordinates_counter = {}
for line in search_file:
    nb_line += 1
    print('line number :', nb_line, ' & time:', time.time() - start, end="\r")
    search_components = line.split(";")
    coord_string = "{0:.2f},{1:.2f}".format(float(search_components[5]), float(search_components[4]))
    if coord_string in searches_coordinates_counter:
        searches_coordinates_counter[coord_string] += 1
    else:
        searches_coordinates_counter[coord_string] = 1
                
search_file.close()

visualization_on_map.draw_search_points_on_map(searches_coordinates_counter)


visualization_on_map.draw_agencies_points_on_map("offices_coordinates.csv")

plt.scatter(-1000, -1000, s=3, zorder=1, c=visualization_on_map.VERY_LIGHT_RED, label="< 25 searches")
plt.scatter(-1000, -1000, s=3, zorder=1, c=visualization_on_map.LIGHT_RED, label="25-50 searches")
plt.scatter(-1000, -1000, s=3, zorder=1, c=visualization_on_map.RED, label="50-100 searches")
plt.scatter(-1000, -1000, s=3, zorder=1, c=visualization_on_map.DARK_RED, label="100-250 searches")
plt.scatter(-1000, -1000, s=3, zorder=1, c="#000000", label=">250 searches")
plt.scatter(-1000, -1000, s=3, zorder=2, c="#28fe4a",label="agency")
plt.legend(loc='lower left',prop={'size': 6}, markerscale = 2)

plt.ylabel('latitude')
plt.xlabel('longitude')
plt.imshow(img, extent=(left, right, bottom, up), zorder=0)
plt.xlim(left, right)
plt.ylim(bottom, up)

plt.savefig("search-agencies_map", dpi=1000)