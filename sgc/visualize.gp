set terminal png transparent nocrop enhanced font arial 8 size 1000, 1000 
set output 'heatmaps.png'
unset key
set view map
set style data linespoints
set xtics border in scale 0,0 mirror norotate  offset character 0, 0, 0
set ytics border in scale 0,0 mirror norotate  offset character 0, 0, 0
set ztics border in scale 0,0 nomirror norotate  offset character 0, 0, 0
set nocbtics
set title "Heat Map generated by 'plot' from a stream of XYZ values\nNB: Rows must be separated by blank lines!" 
set rrange [ * : * ] noreverse nowriteback  # (currently [8.98847e+307:-8.98847e+307] )
set trange [ * : * ] noreverse nowriteback  # (currently [-5.00000:5.00000] )
set urange [ * : * ] noreverse nowriteback  # (currently [-5.00000:5.00000] )
set vrange [ * : * ] noreverse nowriteback  # (currently [-5.00000:5.00000] )
set xrange [ -0.5 : * ] noreverse nowriteback
set x2range [ * : * ] noreverse nowriteback  # (currently [-0.500000:4.50000] )
set yrange [ -0.5 : * ] noreverse nowriteback
set y2range [ * : * ] noreverse nowriteback  # (currently [-0.500000:4.50000] )
set zrange [ 0.0 : 1.0 ] noreverse nowriteback  # (currently [0.00000:5.00000] )
set cblabel "Score" 
set cbrange [ 0.00000 : * ] noreverse nowriteback
set palette rgbformulae -7, 2, -7
plot 'result_sorted_final.csv' using 2:1:3 with image