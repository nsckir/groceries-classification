#!/bin/bash


# This script assumes the following structure  in ../../data/raw/scoodit_178/
#  /label1/image0.jpeg
#  /label1/image1.jpg
#  /label1/image2.png
#  ...
#  /label2/weird-image.jpeg
#  /label2/my-image.jpeg
#  /label2/my-image.JPG

cd ../../data/raw/scoodit_178/
mkdir -p train;
mkdir -p test; 
for i in ./*; do 
	c=`basename "$i"`; 
	fraction=0.95
	mkdir -p train/"$c";
	mkdir -p test/"$c"; 
	count=$(find "$i"/*.JPEG -maxdepth 1 -type f|wc -l); 
	tr_samples=$(expr ${count}*${fraction} | bc)
	round_tr_samples=${tr_samples%.*}
	echo "$c"
	echo ${count};
	echo ${round_tr_samples};
    for j in `ls "$i"/*.JPEG | shuf | head -n ${round_tr_samples}`; do
        mv "$j" train/"$c"/
    done
    mv "$i"/*.JPEG test/"$c"/
done