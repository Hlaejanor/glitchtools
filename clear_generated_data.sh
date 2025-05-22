#!/bin/sh
echo pwd
echo 'Clearing generated data'
rm ./fits/synthetic_lightlane/*_gen.fits

echo 'Clearing generated meta'
rm ./meta_files/fits/meta_*.json
rm ./meta_files/generation/*_gen.jsoni


