#!/bin/bash

cd ../../data/raw/scoodit_178/
mkdir -p train;
mkdir -p test; 
for i in ./*; do 
	c=`basename "$i"`; 
	frac=0.95
	mkdir -p train/"$c";
	mkdir -p test/"$c"; 
	count=$(find "$i"/*.JPEG -maxdepth 1 -type f|wc -l); 
	tr_samples=$(expr $count*$frac | bc)
	round_tr_samples=${tr_samples%.*}
	echo "$c"
	echo $count; 
	echo $round_tr_samples; 
    for j in `ls "$i"/*.JPEG | shuf | head -n $round_tr_samples`; do
        mv "$j" train/"$c"/
    done
    mv "$i"/*.JPEG test/"$c"/
done