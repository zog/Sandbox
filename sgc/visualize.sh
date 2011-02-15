#! /bin/sh
# visualize.sh

cat result.csv|sort -n -k1 -k2 > result_sorted.csv
awk '{ print;if ((NR % 272) == 0)  printf("\n");}' result_sorted.csv > result_sorted_final.csv 
gnuplot visualize.gp ; open heatmaps.png