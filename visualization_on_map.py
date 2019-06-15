import pandas
import random
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cbook
import statistics




def draw_search_points_on_map(search_dictionary):
    for key, value in search_dictionary.items():
            coord = key.split(",")
            if value > 500:
                colour = "#000000"
            elif value > 250:
                colour = VERY_DARK_RED
            elif value > 100:
                colour = DARK_RED
            elif value > 50:
                colour = RED
            elif value > 25:
                colour = LIGHT_RED
            else:
                colour = VERY_LIGHT_RED
            plt.scatter(float(coord[0]), float(coord[1]), s=3, zorder=1, c=colour)

def draw_agencies_points_on_map(filename):
    file = open("offices_coordinate.csv")
    line = file.readline()
    for line in file:
        tmp = line.split(";")
        if(tmp[0] != "NULL" and tmp[1] != "NULL"):
            plt.scatter(float(tmp[0]), float(tmp[1]), s=3, zorder=2,c="#28fe4a")
    file.close()

OUTPUT_DIRECTORY = "outputs\\"

VERY_LIGHT_RED = "#ffbbbb"
LIGHT_RED = "#ff6060"
RED = "#ff0000"
DARK_RED = "#dd0000"
VERY_DARK_RED = "#550000"

if __name__ == "__main__":

    datafile = cbook.get_sample_data('C:\\Users\\P62116\\Documents\\Code\\map_gris.png')
    img = plt.imread(datafile)

    left = 2+7/24
    right = 6+11/18
    bottom = 49+1/3
    up = 51+7/12

    filename = sys.argv[1]

    f = open(filename, "r")
    clustering_algorithm = filename.split("-")[1].split(".")[0]
    label = 0
    counter = 0
    fig_counter = 1
    start = time.time()
    plt.figure(fig_counter)
    search_per_coord_counter = {}
    budget_max =  []
    nb_bedroom =  []
    propertyType = []
    transferType = []
    for line in f:
        counter += 1
        print('line number :', counter, ' & time:', time.time() - start, end="\r")
        tmp = line.split(";")
        if label != int(float(tmp[6])):
            draw_search_points_on_map(search_per_coord_counter)

            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.xlim(left, right)
            plt.ylim(bottom, up)
            plt.imshow(img, extent=(left, right, bottom, up), zorder=0)
            plt.scatter(-1000, -1000, s=3, zorder=1, c=VERY_LIGHT_RED, label="< 25 searches")
            plt.scatter(-1000, -1000, s=3, zorder=1, c=LIGHT_RED, label="25-50 searches")
            plt.scatter(-1000, -1000, s=3, zorder=1, c=RED, label="50-100 searches")
            plt.scatter(-1000, -1000, s=3, zorder=1, c=DARK_RED, label=">100 searches")

            plt.legend(loc='lower left',prop={'size': 6}, markerscale = 2)
            plt.title("propertyType: "+str(round(statistics.mean(propertyType),2))+"transferType: "+str(round(statistics.mean(transferType),2))+"price : "+str(round(statistics.mean(budget_max)))+" - bedroom nb : "+str(round(statistics.mean(nb_bedroom))))

            plt.savefig(OUTPUT_DIRECTORY+clustering_algorithm+"-label"+str(fig_counter), dpi=1000)

            fig_counter += 1
            plt.figure(fig_counter)
            search_per_coord_counter = {}
            label = int(float(tmp[6]))
            nb_bedroom = []
            budget_max = []
            propertyType = []
            transferType = []
        coord_string = "{0:.2f},{1:.2f}".format(
            float(tmp[4])*100, float(tmp[3])*100)
        if(str.isdigit(tmp[0].split(".")[0])):
            propertyType.append(float(tmp[0].split(".")[0]))
        if(str.isdigit(tmp[1].split(".")[0])):
            transferType.append(float(tmp[1].split(".")[0]))
        if(str.isdigit(tmp[2].split(".")[0])):
            budget_max.append(float(tmp[2].split(".")[0]))
        if(str.isdigit(tmp[5].split(".")[0])):
            nb_bedroom.append(int(tmp[5].split(".")[0]))

        if coord_string in search_per_coord_counter:
            search_per_coord_counter[coord_string] += 1
        else:
            search_per_coord_counter[coord_string] = 1
    draw_search_points_on_map(search_per_coord_counter)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.xlim(left, right)
    plt.ylim(bottom, up)
    plt.imshow(img, extent=(left, right, bottom, up), zorder=0)

    plt.scatter(-1000, -1000, s=3, zorder=1, c=VERY_LIGHT_RED, label="< 25 searches")
    plt.scatter(-1000, -1000, s=3, zorder=1, c=LIGHT_RED, label="25-50 searches")
    plt.scatter(-1000, -1000, s=3, zorder=1, c=RED, label="50-100 searches")
    plt.scatter(-1000, -1000, s=3, zorder=1, c=DARK_RED, label=">100 searches")

    plt.legend(loc='lower left',prop={'size': 6}, markerscale = 2)
    plt.title("property type: "+str(statistics.mean(propertyType))+"transfer type: "+str(statistics.mean(transferType))+"budget max: "+str(round(statistics.mean(budget_max)))+" - bedroom number : "+str(round(statistics.mean(nb_bedroom))))
    plt.savefig(OUTPUT_DIRECTORY+clustering_algorithm+"-label"+str(fig_counter), dpi=1000)
    print('line number :', counter, ' & time:', time.time() - start, end="\n")
    f.close()
