#!/bin/bash

# Script to perform measurements for the GstSeamCrop element.

################################################
# DASH Streaming experiments
################################################



# ---- Scenario 1 ----



#############################
## Retargeting factor: 0.85 #
#############################

#top -d 1 -b -n 280 > raw_data/TOP_1_1_1.txt &
#./evaluationseamcrop --no-sync --measure --dynamic --retarget 0.85 --size 50 > raw_data/1_1_1.txt
#sleep 20

#echo "Finished 1_1_1"

#top -d 1 -b -n 280 > raw_data/TOP_1_1_2.txt &
#./evaluationseamcrop --no-sync --measure --dynamic --retarget 0.85 --size 100 > raw_data/1_1_2.txt
#sleep 20
#echo "Finished 1_1_2"

#top -d 1 -b -n 280 > raw_data/TOP_1_1_3.txt &
#./evaluationseamcrop --no-sync --measure --dynamic --retarget 0.85 --size 200 > raw_data/1_1_3.txt
#sleep 20
#echo "Finished 1_1_3"

#############################
## Retargeting factor: 0.75 #
#############################

#top -d 1 -b -n 280 > raw_data/TOP_1_2_1.txt &
#./evaluationseamcrop --no-sync --measure --dynamic --retarget 0.75 --size 50 > raw_data/1_2_1.txt
#sleep 20
#echo "Finished 1_2_1"

#top -d 1 -b -n 280 > raw_data/TOP_1_2_2.txt &
#./evaluationseamcrop --no-sync --measure --dynamic --retarget 0.75 --size 100 > raw_data/1_2_2.txt
#sleep 20
#echo "Finished 1_2_2"

#top -d 1 -b -n 280 > raw_data/TOP_1_2_3.txt &
#./evaluationseamcrop --no-sync --measure --dynamic --retarget 0.75 --size 200 > raw_data/1_2_3.txt
#sleep 20
#echo "Finished 1_2_3"

#############################
## Retargeting factor: 0.50 #
#############################

#top -d 1 -b -n 280 > raw_data/TOP_1_3_1.txt &
#./evaluationseamcrop --no-sync --measure --dynamic --retarget 0.50 --size 50 > raw_data/1_3_1.txt
#sleep 20
#echo "Finished 1_3_1"

#top -d 1 -b -n 280 > raw_data/TOP_1_3_2.txt &
#./evaluationseamcrop --no-sync --measure --dynamic --retarget 0.50 --size 100 > raw_data/1_3_2.txt
#sleep 20
#echo "Finished 1_3_2"

#top -d 1 -b -n 280 > raw_data/TOP_1_3_3.txt &
#./evaluationseamcrop --no-sync --measure --dynamic --retarget 0.50 --size 200 > raw_data/1_3_3.txt
#sleep 20
#echo "Finished 1_3_3"



# ---- Scenario 2 ----



#############################
## Retargeting factor: 0.85 #
#############################

#top -d 1 -b -n 280 > raw_data/TOP_2_1_1.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.85 --size 50 --frequency 90 --runs 3 > raw_data/2_1_1.txt
#sleep 20

#echo "Finished 2_1_1"

#top -d 1 -b -n 280 > raw_data/TOP_2_1_2.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.85 --size 100 --frequency 90 --runs 3 > raw_data/2_1_2.txt
#sleep 20

#echo "Finished 2_1_2"

#top -d 1 -b -n 280 > raw_data/TOP_2_1_3.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.85 --size 200 --frequency 90 --runs 3 > raw_data/2_1_3.txt
#sleep 20

#echo "Finished 2_1_3"

#############################
## Retargeting factor: 0.75 #
#############################

#top -d 1 -b -n 280 > raw_data/TOP_2_2_1.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.75 --size 50 --frequency 90 --runs 3 > raw_data/2_2_1.txt
#sleep 20

#echo "Finished 2_2_1"

#top -d 1 -b -n 280 > raw_data/TOP_2_2_2.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.75 --size 100 --frequency 90 --runs 3 > raw_data/2_2_2.txt
#sleep 20

#echo "Finished 2_2_2"

#top -d 1 -b -n 280 > raw_data/TOP_2_2_3.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.75 --size 200 --frequency 90 --runs 3 > raw_data/2_2_3.txt
#sleep 20

#echo "Finished 2_2_3"


#############################
## Retargeting factor: 0.50 #
#############################

#top -d 1 -b -n 280 > raw_data/TOP_2_3_1.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.50 --size 50 --frequency 90 --runs 3 > raw_data/2_3_1.txt
#sleep 20

#echo "Finished 2_3_1"

#top -d 1 -b -n 280 > raw_data/TOP_2_3_2.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.50 --size 100 --frequency 90 --runs 3 > raw_data/2_3_2.txt
#sleep 20

#echo "Finished 2_3_2"

#top -d 1 -b -n 300 > raw_data/TOP_2_3_3.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.50 --size 200 --frequency 90 --runs 3 > raw_data/2_3_3.txt
#sleep 20

#echo "Finished 2_3_3"



#----- Scenario 3 -----



#############################
## Retargeting factor: 0.85 #
#############################

#top -d 1 -b -n 280 > raw_data/TOP_3_1_1.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.85 --size 50 --frequency 30 --runs 9 > raw_data/3_1_1.txt
#sleep 20

#echo "Finished 3_1_1"

#top -d 1 -b -n 280 > raw_data/TOP_3_1_2.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.85 --size 100 --frequency 30 --runs 9 > raw_data/3_1_2.txt
#sleep 20

#echo "Finished 3_1_2"

#top -d 1 -b -n 280 > raw_data/TOP_3_1_3.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.85 --size 200 --frequency 30 --runs 9 > raw_data/3_1_3.txt
#sleep 20

#echo "Finished 3_1_3"

#top -d 1 -b -n 280 > raw_data/TOP_3_2_1.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.75 --size 50 --frequency 30 --runs 9 > raw_data/3_2_1.txt
#sleep 20

#############################
## Retargeting factor: 0.75 #
#############################

#echo "Finished 3_2_1"

#top -d 1 -b -n 280 > raw_data/TOP_3_2_2.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.75 --size 100 --frequency 30 --runs 9 > raw_data/3_2_2.txt
#sleep 20

#echo "Finished 3_2_2"

#top -d 1 -b -n 280 > raw_data/TOP_3_2_3.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.75 --size 200 --frequency 30 --runs 9 > raw_data/3_2_3.txt
#sleep 20

#echo "Finished 3_2_3"

#top -d 1 -b -n 280 > raw_data/TOP_3_3_1.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.50 --size 50 --frequency 30 --runs 9 > raw_data/3_3_1.txt
#sleep 20

#############################
## Retargeting factor: 0.50 #
#############################

#echo "Finished 3_3_1"

#top -d 1 -b -n 280 > raw_data/TOP_3_3_2.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.50 --size 100 --frequency 30 --runs 9 > raw_data/3_3_2.txt
#sleep 20

#echo "Finished 3_3_2"

#top -d 1 -b -n 280 > raw_data/TOP_3_3_3.txt &
#./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure  --retarget 0.50 --size 200 --frequency 30 --runs 9 > raw_data/3_3_3.txt
#sleep 20

#echo "Finished 3_3_3"



# --- Initial Latency Measurements ---



#for i in `seq 1 10`;
#do
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure --retarget 0.85 --size 50 --runs 1 --frequency 10 >> raw_data/4_1_1.txt
#sleep 1
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=854,height=480 --no-sync --measure --retarget 0.85 --size 50 --runs 1 --frequency 10 >> raw_data/4_1_1.txt
#sleep 1  
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=1280,height=720 --no-sync --measure --retarget 0.85 --size 50 --runs 1 --frequency 10 >> raw_data/4_1_1.txt
#sleep 1
#done

#echo "Finished 4_1_1.txt"

#for i in `seq 1 10`;
#do
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure --retarget 0.85 --size 100 --runs 1 --frequency 10 >> raw_data/4_1_2.txt
#sleep 1
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=854,height=480 --no-sync --measure --retarget 0.85 --size 100 --runs 1 --frequency 10 >> raw_data/4_1_2.txt
#sleep 1  
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=1280,height=720 --no-sync --measure --retarget 0.85 --size 100 --runs 1 --frequency 10 >> raw_data/4_1_2.txt
#sleep 1
#done

#echo "Finished 4_1_2.txt"

#for i in `seq 1 10`;
#do
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure --retarget 0.85 --size 200 --runs 1 --frequency 10 >> raw_data/4_1_3.txt
#sleep 1
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=854,height=480 --no-sync --measure --retarget 0.85 --size 200 --runs 1 --frequency 10 >> raw_data/4_1_3.txt
#sleep 1  
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=1280,height=720 --no-sync --measure --retarget 0.85 --size 200 --runs 1 --frequency 10 >> raw_data/4_1_3.txt
#sleep 1
#done

#echo "Finished 4_1_3.txt"


#for i in `seq 1 10`;
#do
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure --retarget 0.75 --size 50 --runs 1 --frequency 10 >> raw_data/4_2_1.txt
#sleep 1
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=854,height=480 --no-sync --measure --retarget 0.75 --size 50 --runs 1 --frequency 10 >> raw_data/4_2_1.txt
#sleep 1  
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=1280,height=720 --no-sync --measure --retarget 0.75 --size 50 --runs 1 --frequency 10 >> raw_data/4_2_1.txt
#sleep 1
#done

#echo "Finished 4_2_1.txt"

#for i in `seq 1 10`;
#do
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure --retarget 0.75 --size 100 --runs 1 --frequency 10 >> raw_data/4_2_2.txt
#sleep 1
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=854,height=480 --no-sync --measure --retarget 0.75 --size 100 --runs 1 --frequency 10 >> raw_data/4_2_2.txt
#sleep 1  
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=1280,height=720 --no-sync --measure --retarget 0.75 --size 100 --runs 1 --frequency 10 >> raw_data/4_2_2.txt
#sleep 1
#done

#echo "Finished 4_2_2.txt"

#for i in `seq 1 10`;
#do
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure --retarget 0.75 --size 200 --runs 1 --frequency 10 >> raw_data/4_2_3.txt
#sleep 1
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=854,height=480 --no-sync --measure --retarget 0.75 --size 200 --runs 1 --frequency 10 >> raw_data/4_2_3.txt
#sleep 1  
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=1280,height=720 --no-sync --measure --retarget 0.75 --size 200 --runs 1 --frequency 10 >> raw_data/4_2_3.txt
#sleep 1
#done

#echo "Finished 4_2_3.txt"

#for i in `seq 1 10`;
#do
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure --retarget 0.50 --size 50 --runs 1 --frequency 10 >> raw_data/4_3_1.txt
#sleep 1
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=854,height=480 --no-sync --measure --retarget 0.50 --size 50 --runs 1 --frequency 10 >> raw_data/4_3_1.txt
#sleep 1  
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=1280,height=720 --no-sync --measure --retarget 0.50 --size 50 --runs 1 --frequency 10 >> raw_data/4_3_1.txt
#sleep 1
#done

#echo "Finished 4_3_1.txt"

#for i in `seq 1 2`;
#do
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure --retarget 0.50 --size 100 --runs 1 --frequency 10 >> raw_data/4_3_2.txt
#sleep 1
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=854,height=480 --no-sync --measure --retarget 0.50 --size 100 --runs 1 --frequency 10 >> raw_data/4_3_2.txt
#sleep 1  
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=1280,height=720 --no-sync --measure --retarget 0.50 --size 100 --runs 1 --frequency 10 >> raw_data/4_3_2.txt
#sleep 1
#done

#echo "Finished 4_3_2.txt"

#for i in `seq 1 10`;
#do
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=640,height=360 --no-sync --measure --retarget 0.50 --size 200 --runs 1 --frequency 10 >> raw_data/4_3_3.txt
#sleep 1
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=854,height=480 --no-sync --measure --retarget 0.50 --size 200 --runs 1 --frequency 10 >> raw_data/4_3_3.txt
#sleep 1  
#  ./evaluationseamcrop --uri http://dash.edgesuite.net/dash264/TestCases/2c/qualcomm/2/MultiRes.mpd --scale video/x-raw,width=1280,height=720 --no-sync --measure --retarget 0.50 --size 200 --runs 1 --frequency 10 >> raw_data/4_3_3.txt
#sleep 1
#done

#echo "Finished 4_3_3.txt"


##################################################################
# HLS Streaming experiments
##################################################################

res_w[0]=640
res_w[1]=854
res_w[2]=1280

res_h[0]=360
res_h[1]=480
res_h[2]=720

#############################
## Retargeting factor: 0.85 #
#############################

for i in `seq 0 2`;
do
  ./evaluationseamcrop --uri http://hlsbook.net/wp-content/examples/sintel/sintel_index.m3u8 --scale video/x-raw,width=${res_w[i]},height=${res_h[i]} --no-sync --measure  --retarget 0.85 --size 50 --frequency 90 --runs 1 >> raw_data/5_1_1.txt
done

echo "Finished 5_1_1.txt"

for i in `seq 0 2`;
do
  ./evaluationseamcrop --uri http://hlsbook.net/wp-content/examples/sintel/sintel_index.m3u8 --scale video/x-raw,width=${res_w[i]},height=${res_h[i]} --no-sync --measure  --retarget 0.85 --size 100 --frequency 90 --runs 1 >> raw_data/5_1_2.txt
done

echo "Finished 5_1_2.txt"


for i in `seq 0 2`;
do
  ./evaluationseamcrop --uri http://hlsbook.net/wp-content/examples/sintel/sintel_index.m3u8 --scale video/x-raw,width=${res_w[i]},height=${res_h[i]} --no-sync --measure  --retarget 0.85 --size 200 --frequency 90 --runs 1 >> raw_data/5_1_3.txt
done

echo "Finished 5_1_3.txt"

#############################
## Retargeting factor: 0.75 #
#############################

for i in `seq 0 2`;
do
  ./evaluationseamcrop --uri http://hlsbook.net/wp-content/examples/sintel/sintel_index.m3u8 --scale video/x-raw,width=${res_w[i]},height=${res_h[i]} --no-sync --measure  --retarget 0.75 --size 50 --frequency 90 --runs 1 >> raw_data/5_2_1.txt
done

echo "Finished 5_2_1.txt"

for i in `seq 0 2`;
do
  ./evaluationseamcrop --uri http://hlsbook.net/wp-content/examples/sintel/sintel_index.m3u8 --scale video/x-raw,width=${res_w[i]},height=${res_h[i]} --no-sync --measure  --retarget 0.75 --size 100 --frequency 90 --runs 1 >> raw_data/5_2_2.txt
done

echo "Finished 5_2_2.txt"


for i in `seq 0 2`;
do
  ./evaluationseamcrop --uri http://hlsbook.net/wp-content/examples/sintel/sintel_index.m3u8 --scale video/x-raw,width=${res_w[i]},height=${res_h[i]} --no-sync --measure  --retarget 0.75 --size 200 --frequency 90 --runs 1 >> raw_data/5_2_3.txt
done

echo "Finished 5_2_3.txt"


#############################
## Retargeting factor: 0.50 #
#############################

for i in `seq 0 2`;
do
  ./evaluationseamcrop --uri http://hlsbook.net/wp-content/examples/sintel/sintel_index.m3u8 --scale video/x-raw,width=${res_w[i]},height=${res_h[i]} --no-sync --measure  --retarget 0.50 --size 50 --frequency 90 --runs 1 >> raw_data/5_3_1.txt
done

echo "Finished 5_3_1.txt"

for i in `seq 0 2`;
do
  ./evaluationseamcrop --uri http://hlsbook.net/wp-content/examples/sintel/sintel_index.m3u8 --scale video/x-raw,width=${res_w[i]},height=${res_h[i]} --no-sync --measure  --retarget 0.50 --size 100 --frequency 90 --runs 1 >> raw_data/5_3_2.txt
done

echo "Finished 5_3_2.txt"


for i in `seq 0 2`;
do
  ./evaluationseamcrop --uri http://hlsbook.net/wp-content/examples/sintel/sintel_index.m3u8 --scale video/x-raw,width=${res_w[i]},height=${res_h[i]} --no-sync --measure  --retarget 0.50 --size 200 --frequency 90 --runs 1 >> raw_data/5_3_3.txt
done

echo "Finished 5_3_3.txt"
