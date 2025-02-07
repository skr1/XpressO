# XpressO

## About

In this study, we collected diagnostic WSIs for IBC tissue biopsies from 191 patients from The
Cancer Genome Atlas (TCGA) [web-portal](https://portal.gdc.cancer.gov/projects/TCGA-BRCA). RNA-seq data for each of the 191 WSIs were
downloaded in the form of fragments per kilobase per million reads (FPKM) values from the
same portal. The DL-based weakly supervised algorithm Clustering-constrained Attention
Multiple Instance Learning (CLAM) was employed to segment tumor regions of interest (ROIs)
on the WSIs. CLAM leverages attention mechanisms to automatically detect sub-
regions with high diagnostic significance for accurate whole-slide classification. Additionally, it
employs instance-level clustering on the identified representative regions to constrain and refine
the feature space, enhancing model interpretability, and performance.
As scanned WSIs result in large files and high-resolution images, they were divided into smaller
and more manageable patches (~1000 patches per WSI) using the [OpenCV](https://opencv.org/) module in Python. Using a sliding window approach, the ROI mask guided the extraction of fixed-size square patches (e.g., 256 × 256 pixels) at the desired magnification level of 20x. To extract features
from these masks, we utilized a pre-trained Unified Network for Instance-level Representation
Learning (UNI) model. The pre-trained UNI model is a weakly supervised Vision
Transformer (ViT-L/16) via DINOv2 optimized for histopathological image analysis,
enabling robust extraction of high-dimensional features (i.e., feature embeddings from WSI
patches). The resulting feature embeddings offered a computationally efficient and accurate
representation of histopathological patterns that were fed into the downstream task of
classification of gene expression for the WSIs.

### Installation

To install the pre-requisite libraries, we need to install Anaconda, more information about downloading and setting up Anaconda can be found [here](https://docs.anaconda.com/anaconda/install/)

Once Anaconda is installed, follow these steps to install the libraries. We have currently run all our simulations on Python 3.10.

`conda env create -f env.yml`
`conda activate xpresso_env`

Clone the GitHub repository

`git clone https://github.com/skr1/XpressO.git`

Installing the UNI model requires access requests, and the steps to follow for proper installation are below.

1. Register for the UNI model through this [link](https://huggingface.co/MahmoodLab/UNI)
2. Create an account in HuggingFace, if you haven't already, and then proceed with adding ssh and gpg keys following this [tuitorial](https://huggingface.co/docs/hub/security-git-ssh)
3. Run hugging face cli login, and authenticate by adding the token to your profile, and generating tokens.
4. Download the pretrained weights by running the script
   `python3 install_uni.py`.
5. Set checkpoint path using `export UNI_CKPT_PATH=MODEL_PATH` where MODEL_PATH is the location you chose to download the weights.

Now, that we have all these setups done, we are good to go with training a Deep Learning-based Weakly Supervised Algorithm.

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

Once we have the segmented patches with us, we proceed with extracting features using the UNI model which is a Vision Transformer. 

To utilize effective GPU utilization while running the feature extraction algorithm, run this command below.
`export CUDA_VISIBLE_DEVICES=0`

Run, the algorithm following the steps below. To leverage the UNI model, you would have to specify the `model-name` parameter, supported algorithms include uni_v1, resnet-50, and conch_v1.

`CUDA_VISIBLE_DEVICES=0 python3 extract_features_fp.py --data_h5_dir BREAST_CANCER_PATCHES --data_slide_dir ORIGINAL_SLIDES_FOLDER --csv_path BREAST_CANCER_PATCHES/process_list_autogen.csv –model-name=”uni_v1” --feat_dir tcga_breast_extracted_features --batch_size 512 --slide_ext .svs`

The resultant file tcga_breast_extracted_features, looks like this
```
tcga_breast_extracted_features/
    ├── h5_files
            ├── slide_1.h5
            ├── slide_2.h5
            └── ...
    └── pt_files
            ├── slide_1.pt
            ├── slide_2.pt
            └── ...
```

## Classification

The classification part of this project involves creating datasets for low-expression, and high-expression genes, and we leverage FPKM gene expression tables for creating relevant datasets. To proceed with this, we run the commands below.

`python3 gene_file_creater_clam.py --csv_file Supplementary_Table_BC-Gene_FPKM.txt --column_name BIOMARKER_NAME --h5_source_folder tcga_breast_extracted_features/h5_files --pt_source_folder tcga_breast_extracted_features/pt_files --low_h5_folder GENE_EXP_CLAM/low_expression_genes/h5_files --low_pt_folder GENE_EXP_CLAM/low_expression_genes/pt_files --high_h5_folder GENE_EXP_CLAM/high_expression_genes/h5_files --high_pt_folder GENE_EXP_CLAM/high_expression_genes/pt_files`

Now, we create CSV files with patient_id, slide_id, and labels based on the requirements to train the CLAM SB model. 

## Grad CAM
