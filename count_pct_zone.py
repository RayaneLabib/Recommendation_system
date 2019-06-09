import sys

VLAAMS_BRABANT_MIN_LAT =0.5075
VLAAMS_BRABANT_MAX_LAT =0.51
VLAAMS_BRABANT_MIN_LON =0.0415
VLAAMS_BRABANT_MAX_LON =0.05

LIMB_MIN_LAT =0.51
LIMB_MAX_LAT =0.515
LIMB_MIN_LON =0.0415
LIMB_MAX_LON =0.0515


if __name__ == "__main__":
    cluster_file = open(sys.argv[1])

    count =0
    nb_line = 0
    cluster_nb = float(sys.argv[2])
    for line in cluster_file:
        split_line = line.split(";")
        if float(split_line[6]) == cluster_nb:
            nb_line += 1
            if VLAAMS_BRABANT_MIN_LAT < float(split_line[3]) < VLAAMS_BRABANT_MAX_LAT and VLAAMS_BRABANT_MIN_LON < float(split_line[4]) < VLAAMS_BRABANT_MAX_LON:
                count +=1
    print("count : ",count/nb_line)
    print("total : ",nb_line)

    cluster_file.seek(0)
    count =0
    nb_line = 0
    for line in cluster_file:
        split_line = line.split(";")
        if float(split_line[6]) == cluster_nb:
            nb_line += 1
            if LIMB_MIN_LAT < float(split_line[3]) < LIMB_MAX_LAT and LIMB_MIN_LON < float(split_line[4]) < LIMB_MAX_LON:
                count +=1
    print("count : ",count/nb_line)
    print("total : ",nb_line)

    cluster_file.close()