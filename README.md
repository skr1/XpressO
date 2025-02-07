# XpressO

## About

This project aims to segment regions of interest using TCGA coordinates and extract features using a pretrained Vision Transformer (UNI). We then proceed with identifying low and high gene expressions and train a UNI model to classify low/high gene expressions through Grad CAM or gradient class activation maps.

### Installation

To install the pre-requisite libraries, we need to install Anaconda, more information about downloading and setting up Anaconda can be found [here](https://docs.anaconda.com/anaconda/install/)

Once Anaconda is installed, follow these steps to install the libraries. We have currently run all our simulations on Python 3.10.

`conda env create -f env.yml`
`conda activate xpresso_env`

Clone the GitHub repository

`git clone https://github.com/skr1/XpressO.git`

We sourced the datasets from https://portal.gdc.cancer.gov/projects/TCGA-BRCA, and our dataset consists of breast TCGA slides.

## Segmentation

Once we have the slides, and installed the pre-requisite libraries, we proceed with segmenting regions of interest using tcga coordinates available in the presets folder under the Segmentation folder. We borrowed implementational guidelines from this [repo](https://github.com/mahmoodlab/CLAM).

```
ORIGINAL_SLIDES_FOLDER/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

`python3 create_patches_fp.py --source ORIGINAL_SLIDES_FOLDER --save_dir BREAST_CANCER_PATCHES --patch_size 256 --preset tcga.csv --seg --patch --stitch`

```
BREAST_CANCER_PATCHES/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```
 
--seg parameter indicates that we want to segment the images based on the preset argument applied. --patch parameter is false by default, so to create patches, we pass it as an argument while creating patches. The patches are created based on patch size parameters, and patch level.

## Feature Extraction

## Classification

## Grad CAM
