#!/bin/sh

echo "Copying antares files"
mkdir /Users/jenstandstad/My\ Drive/Atlantean/Latex/Undergraduate\ Physics/final/plots
mkdir  /Users/jenstandstad/My\ Drive/Atlantean/Latex/Undergraduate\ Physics/final/tables

cp -R plots/* /Users/jenstandstad/My\ Drive/Atlantean/Latex/Undergraduate\ Physics/final/plots
cp -r tables/*.tex /Users/jenstandstad/My\ Drive/Atlantean/Latex/Undergraduate\ Physics/final/tables/
echo "copied files"
