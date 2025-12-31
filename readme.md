# Lightlane Studio v1.0

**License Notice**  
You are free to study, review, and discuss this code. However, if you intend to use it in a commercial setting or for teaching purposes, you must obtain a license. See Licence.md

## Purpose

This project serves three goals:

1. **Scientific Analysis**  
   To generate plots used in the book *"Atomism 2.0"*, which investigates flux variability in the Cosmic X-ray Background (CXB), particularly in the 0.1–1 nm range.

2. **Model Testing**  
   To create mimic datasets that contain the flux variability of the Lightlane variety. These are used as controls in the analysis pipeline

3. **Visualization**  
   To visualize a telescope moving across a "Lanesheet" - a conceptual grid of lightlanes - to aid in understanding the model’s dynamics.


### Installation instructions

# Step 1. Clone the repository
`git clone https://github.com/lightlaneinstitute/lightlane-pipeline.git
`cd lightlane-pipeline
# Step 2. Create an environment (Python 3.10+ recommended)
`python3 -m venv venv
`source venv/bin/activate
`pip install -r requirements.txt
# Step 3. Download datasets (optional if included)


---
## Tools and utilities: 
This repo has three main tools, listed below
0. run_all_pipelines.sh -> Runs all pipelines by calling the json files in meta/pipeline. A pipeline is a comparison over two datasets using the same Processing Parameters. 
1. make_var_analysis_plots.sh -> Used to generate plots that visualize the variability 
2. create_pipeline.sh -> Used to create a pipeline, specify filters to run
3. make_lanesheet_video.sh -> Used to make a video of a synthetic (or real!) dataset
4. parameter_search.sh -> Used to generate datasets that and find the right parameters for 


## Filters in 
create_pipeline.sh 

### 'generate' filter
In stead of using an existing dataset, use the GenerationParameter to generate a dataset

### 'chandrashift' filter
Shift the time from the n-th slot to the n minus 1-th slot and dicard the first and last observation in the dataset.

Reference: https://cxc.harvard.edu/proposer/POG/html/chap7.html#sec:hrc_anom
A wiring error in the HRC causes the time of an event to be associated with the following event,
which may or may not be telemetered. The result is an error in HRC event timing that degrades accuracy from about 16 microsec to roughly the mean time between events. For example, if the trigger rate is 250 events/sec, then the average uncertainty in any time tag is less than 4 millisec.


### 'wbin' filter
Apply the Wavelength Bin column by cutting the data using wavelength bins from ProcessingParameters

### 'wrange' filter
Use the Wavelength (nm) column and include only events that are in the range max_wavelength, min_wavelength (given in ProcessingParameters)

### 'flatten' filter
Select the count from the smallest wbin in the range, and downsample all other wbins so they match the count

### 'poissonize' filter
Randomize the time column of the events

### 'downsample' filter
Downsample the dataset to a smaller number of observations

### 'equalize' filter
Make sure that all wbins have the same count. Done by sampling from within the bin population (random duplication)

### 'copysource'
Copy the source dataset into the slot

### 'copyA'
Only on B -slot : Use the output from the A pipe as input for the B pipeline


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

´/create_default_fits_metadata.sh --file ./fits/15734/primary/hrcf15734N003_evt2.fits.gz -n antares

6. Create the default pipeline. This will create the default Processing Pipeline, Generation Parameters and the Compare pipeline
´./create_default_pipeline.sh´

7. You are now ready to run the various analysis files. This can be done in two ways, by running a pipeline which will select the datasets for you, or by using the tools

## Generating plots for a single dataset (only)
To analyze and plot data from the real Chandra dataset, you may use:

´make_var_analysis.sh -a antares --pp default´

This references the dataset described by meta/fits/antares.json, the dataset downloaded in step 5 

# A - B testing and for two datasets
The pipeline is set up to generate plots for two datasets, A and B, where A is the real data from chandra and B is a synthetic dataset generated using a set of GenerationParameters. The pipe connects A and B and also provides parameters for binning, filtering etc in ProcessingParams.

1. A good first step is to generate a synthetic dataset using the default pipeline. 
´replicate_dataset.sh -n antares_vs_anisogen`

2. Inspect the default_pipe.json. Notice it now references a new B_fits_id code. This 8 letter code is the name of the synthetic dataset. 

3. To compare the A and B datasets referenced in A_vs_B.json you can run the 

´make_var_analysis.sh -p antares_vs_anisogen

This will use the data in meta/pipeline/antares_vs_anisogen.json to generate plots and compare the two datasets A and B.

# Mimic Mode: Variability Replication
The second purpose of Lightlane finder is to generate mimic datasets, that replicate the variability observed in the real data. The parameter_search tool will generate synthetic datasets, write them to disk and write the results to a csv file called ./temp/parameter_search.csv. 

This has two main modes. Search mode randomly generates parameters and produces datasets from them. 

´parameter_search.sh -mode search´

Refine mode takes the best randomly generated dataset that matches the target the best, and tries to refine it to make it even better 

´parameter_search.sh -mode refine´

# Generating video (experimental)
This feature can be used to generate a video of a camera passing over a lanesheet. It requries a pipeline, and for the video to make sense it should have
a very limited wavelength range. The video will generate a lanesheet using the midpoint between the min_wavelength and max_wavelength of the ProcessingParameters, so if this is too wide the timing of the events will not match the Lightlanes
