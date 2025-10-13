#!/bin/sh
echo "Pad wavelength bins to ensure similar size"

python src/padding_strategy.py "$@"

