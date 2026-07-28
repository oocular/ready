#!/bin/bash
set -Eeuxo pipefail

#cd ~/ready/data/bash-scripts
#bash resize_mobious_dataset.bash $HOME/Desktop/nystagmus-tracking/datasets/mobious/MOBIOUS

DATASETPATH=$1
cd $DATASETPATH
echo "Start:" `date`
echo $DATASETPATH

#train path
if [ ! -d train ]
then
     mkdir -p train/images && mkdir -p train/masks
     echo "Directories has been created!"
else
     echo "Directory exists"
fi

##TODO create loop to go trough relevant paths
#for d in $DATASETPATH/Masks/*; do
#    # Will print */ if no directories are available
#    echo "$d"
#done


########
#copy mask images
cp -r $DATASETPATH/Masks/1/* $DATASETPATH/train/masks
cp -r $DATASETPATH/Masks/2/* $DATASETPATH/train/masks
cp -r $DATASETPATH/Masks/3/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/4/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/5/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/6/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/7/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/8/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/9/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/10/* $DATASETPATH/train/masks
cp -r $DATASETPATH/Masks/11/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/12/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/13/* $DATASETPATH/train/masks
cp -r $DATASETPATH/Masks/14/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/15/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/15/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/16/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/17/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/18/* $DATASETPATH/train/masks
cp -r $DATASETPATH/Masks/19/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/20/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/21/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/22/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/23/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/24/* $DATASETPATH/train/masks
cp -r $DATASETPATH/Masks/25/* $DATASETPATH/train/masks
cp -r $DATASETPATH/Masks/26/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/27/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/28/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/29/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/30/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/31/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/32/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/33/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/34/* $DATASETPATH/train/masks
##diff cp -r $DATASETPATH/Masks/35/* $DATASETPATH/train/masks

########
#copy raw Images and remove bad ones
cp -r $DATASETPATH/Images/1/* $DATASETPATH/train/images
cp -r $DATASETPATH/Images/2/* $DATASETPATH/train/images
cp -r $DATASETPATH/Images/3/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/4/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/5/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/6/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/7/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/8/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/9/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/10/* $DATASETPATH/train/images
cp -r $DATASETPATH/Images/11/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/12/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/13/* $DATASETPATH/train/images
cp -r $DATASETPATH/Images/14/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/15/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/15/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/16/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/17/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/18/* $DATASETPATH/train/images
cp -r $DATASETPATH/Images/19/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/20/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/21/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/22/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/23/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/24/* $DATASETPATH/train/images
cp -r $DATASETPATH/Images/25/* $DATASETPATH/train/images
cp -r $DATASETPATH/Images/26/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/27/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/28/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/29/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/30/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/31/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/32/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/33/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/34/* $DATASETPATH/train/images
##diff cp -r $DATASETPATH/Images/35/* $DATASETPATH/train/images
find $DATASETPATH/train/images -type f -iname "*_bad.jpg" -delete 

##resize images
mogrify -resize 640x400! $DATASETPATH/train/masks/*.png
mogrify -resize 640x400! $DATASETPATH/train/images/*.jpg
echo "End:" `date`
echo "Done"
