# Lightlane Finder v1.0

**License Notice**  
You are free to study, review, and discuss this code. However, if you intend to use it in a commercial setting or for teaching purposes, you must obtain a license.

## Purpose

This project serves three goals:

1. **Scientific Analysis**  
   To generate plots used in the book *"Staring Into the Void"*, which investigates flux variability in the Cosmic X-ray Background (CXB), particularly in the 0.1–1 nm range.

2. **Model Testing**  
   To replicate the observed flux variability using a model that introduces *Lightlane Anisotropies* — directional bias in photon detections.

3. **Visualization**  
   To visualize a telescope moving across a "Lanesheet" — a conceptual grid of lightlanes — to aid in understanding the model’s dynamics.

---
## Tools and utilities: 
This repo has three main tools, listed below
1. make_var_analysis_plots.sh -> Used to generate plots that visualize the variability 
2. make_lanesheet_video.sh -> Used to make a video of a synthetic (or real!) dataset
3. replicate_dataset.sh -> Used to replicate a dataset
4. parameter_search.sh -> Used to generate datasets that and find the right parameters for 
5. lanesheet_fitteer -> Used to 

Utitilies:

1. download_chandra_data -> Used to fetch data from Chaser.Harvard.edu
2. Create_fits_metadata.sh -> Used to onboard data from a telescope using the default processing params
3. Create_default_pipeline.sh -> Used to setup a default pipeline for the dataset
4. clear_generated_data.sh -> Remove generated files but 
## Getting Started

To replicate the analysis, follow these steps. Goal 1 requires only the real Chandra dataset. Goals 2 and 3 involve generating synthetic data and visualizations.

---

### Preparation Steps

1. **Download Chandra Data**

   Go to:  
   [https://cda.harvard.edu/chaser/](https://cda.harvard.edu/chaser/)

   - Search for: `Antares`
   - Click **Add to Retrieval List**
   - Then click **Retrieve Products** and copy the generated URL.

2. **Download FITS Files**

   Use the provided download script with the copied URL:

   ```bash
   ./download_chandra_data.sh https://cdaftp.cfa.harvard.edu/pub/stage/zE7kVpfa/


5. Create metadata for a downloaded FITS file. Use -n default to create the default metadata set:

´/create_default_fits_metadata.sh --file ./fits/15734/primary/hrcf15734N003_evt2.fits.gz -n default´

6. Crate the default pipeline. This will create the default Processing Pipeline, Generation Parameters and the Compare pipeline
´./create_default_pipeline.sh´

7. You are now ready to run the various analysis files.

## Generating plots for the Chandra dataset (only)
To analyze and plot data from the real Chandra dataset, you may use:

´make_var_analysis.sh -a default --pp default´

This references the default meta_default.json, created in step 5 and the default_pparam.json which was created in step 6. 

# A - B testing and for two datasets
The pipeline is set up to generate plots for two datasets, A and B, where A is the real data from chandra and B is a synthetic dataset generated
using a set of GenerationParameters. The pipe connects A and B and also provides parameters for binning, filtering etc in ProcessingParams.

1. A good first step is to generate a synthetic dataset using the default pipeline. 
´replicate_dataset.sh -n default`

2. Inspect the default_pipe.json. Notice it now references a new B_fits_id code. This 8 letter code is the name of the synthetic dataset. 

3. To compare the A and B datasets referenced in default_pipe.json you can run the 

´make_var_analysis.sh -p default´

This will use the data in default_pipe.json to generate plots and compare the two datasets A and B. PLots can be found in

# Mimic Mode: Variability Replication
The second purpose of Lightlane finder is to generate mimic datasets, that replicate the variability observed in the real data. 
The parameter_search tool will generate synthetic datasets, write them to disk and write the results to a csv file called ./temp/parameter_search.csv. 

This has two main modes. Search mode randomly generates parameters and produces datasets from them. 

´parameter_search.sh -mode search´

Refine mode takes the best randomly generated dataset that matches the target the best, and tries to refine it to make it even better 

´parameter_search.sh -mode refine´

# Generating video (experimental)
This feature can be used to generate a video of a camera passing over a lanesheet. It requries a pipeline, and for the video to make sense it should have
a very limited wavelength range. The video will generate a lanesheet using the midpoint between the min_wavelength and max_wavelength of the ProcessingParameters, so if this is too wide the timing of the events will not match the Lightlanes

# Lanefitter (experimental)