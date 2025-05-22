# Variability pipeline

## What this repo does
Purpose:
Generate plots for Book1, about Flux variability in X-ray spectrum

Main:
 - Compute excess variability from Chandra data and generate plots
 - Use exact same pipeline to generate comparison plots from a synthetically generated mimic dataset
 - Generate many synthetic datasets, and train an ML model to choose the parameters that look more similar to the real data

## Example:
The idea behind this repo is to plot excess variability from Chandra side by side by excess variability produced by Lightlanes.
To do this, we develop a pipeline for Chandra data that shows excess variability. Then we generate data with Lightlane anisotropies, and we feed it through the same pipeline with the same setting. If the plots look the same, then the Lightane method of generating variability is able to reproduce the observed variability. 

For example : Take a look at the file plots/plot_1_flickerfit_RV_norm_100s.png, which is  on real chandra data. 
If you can make a file called plot_{something}_flickerfit_RV_norm_100s.png, and those two files look the same - then the reason for this might be that the synthetic process reproduces the variaiblity. If the apparent likeness can't be explained by something else, this is evidence that the universe is discrete and the photons have discrete directional precision. 

## Process
1. Ensure that meta_1.json correspond to the Chandra dataset you want to analyze. Adjust binning settings etc to your liking
2. Run parametersearch.py. This will generate datasets based on param_cobmos. Each dataset gets a unique meta_{id}.json file and a summary written to temp/parameter_search.csv
3. With a few examples generated, you may run readandplotchandra.py. This will generate plots annotated with the experiment id.
4. Leave parameter_search running.py


## What is a beat plot?
The beat plot is the Fourier transform of the timing between randomly sampled events in the dataset, for a wavelength bin
A beat plot is a map over the frequencies encountered in the dataset between hits of the same wavelength. 
So, for example : The beat plot of a uniformly distributed (Poissonian) stream of photon hits, will just be white noise.
However, if there is _periodic_ structure, this will show up as frequency peaks in the beat-plot.
The beat plot (including the position, order and relative ratios of the harmonic peaks) contains information about the underlying Lanesheet.
For this reason, the beatplot is used to estimate parameters for the lanesheet. These are excellent guesses, but require
fine-tuning by other, iterative methods.

## What is a lanestack sieve?
A lanestack is a stack of lanesheets. The idea is that if you can make one lanesheet for one frequency, 
you can also make another for the adjacent wavelength bin. If you can align these lanesheets together and 
align them perfectly, this allows you to create a Moire sieve. 

## How to use this repo

To use this repo, use main.py to generate the datasets, then the beat-plots. 
The beat-plot metadata will accumulate in the file ./beat/beat_metadata.csv, and be used to train
the LanesheetParamEstimator, a ML regressor using Random Forest. 
In main.py, if the fine_tune=True set, the guess made by the LanesheetParamEstimator will be
refined by iterative method using a gradient descent search

You can import methods from this module and use them to fit Lanesheets to real datasets 
by obtaining the best fit Lanesheet Params
