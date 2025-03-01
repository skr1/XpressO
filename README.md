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

Navigate to the Segmentation folder `cd Segmentation`

Once we have the slides, and installed the pre-requisite libraries, we proceed with segmenting regions of interest using TCGA coordinates available in the presets folder under the Segmentation folder. We borrowed implementational guidelines from this [repo](https://github.com/mahmoodlab/CLAM).

```
ORIGINAL_SLIDES_FOLDER/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

`python3 create_patches_fp.py --source ORIGINAL_SLIDES_FOLDER --save_dir BREAST_CANCER_PATCHES --patch_size 256 --preset tcga.csv --seg --patch --stitch`

It took us 40 minutes to create segmented patches for 191 samples.

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

Navigate to the Feature Extraction folder `cd Feature Extraction`

Once we have the segmented patches with us, we proceed with extracting features using the UNI model which is a Vision Transformer. 

To utilize effective GPU utilization while running the feature extraction algorithm, run this command below.

`export CUDA_VISIBLE_DEVICES=0`

Run, the algorithm following the steps below. To leverage the UNI model, you would have to specify the `model_name` parameter, supported algorithms include uni_v1, resnet-50, and conch_v1.

`CUDA_VISIBLE_DEVICES=0 python3 extract_features_fp.py --data_h5_dir BREAST_CANCER_PATCHES --data_slide_dir ORIGINAL_SLIDES_FOLDER --csv_path BREAST_CANCER_PATCHES/process_list_autogen.csv 
--model_name=”uni_v1” --feat_dir tcga_breast_extracted_features --batch_size 512 --slide_ext .svs`

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
feature.txt is the log file generated after the command is run.
## Classification

The classification part of this project involves creating datasets for low-expression, and high-expression genes, and we leverage FPKM gene expression tables for creating relevant datasets. To proceed with this, we run the commands below.

gene_file_creater_clam.py works for tab delimited files and gene_file_creater_clam_v2.py works for comma delimited files.

Supplementary_Table_BC-Gene_FPKM.txt is the 89 samples gene expression file, we run the following command:

`python3 gene_file_creater_clam.py --csv_file Supplementary_Table_BC-Gene_FPKM.txt --column_name BIOMARKER_NAME --h5_source_folder tcga_breast_extracted_features/h5_files --pt_source_folder tcga_breast_extracted_features/pt_files --low_h5_folder GENE_EXP_CLAM/low_expression_genes/h5_files --low_pt_folder GENE_EXP_CLAM/low_expression_genes/pt_files --high_h5_folder GENE_EXP_CLAM/high_expression_genes/h5_files --high_pt_folder GENE_EXP_CLAM/high_expression_genes/pt_files`

We ran the same command for the 102 samples gene expression using file Master_concat_2ndattempt_uniq.txt.
Make sure to change the destination folder for the 102 samples. 

```
GENE_EXP_CLAM/
    ├── low_expression_genes
            ├── h5 files
            ├── pt files
    └── high_expression_genes
            ├── h5 files
            └── pt files
	
```

Now, we merge the two folders using the code python_file_merger.py.
Make necessary changes in the code for the source and destination folders

```
src_folders = ["GENE_EXP_CLAM_1", "GENE_EXP_CLAM_2"]
dest_folder = "GENE_EXP_CLAM_merged"
```
Run 
`python3 python_file_merger.py`

Now, we create CSV files with patient_id, slide_id, and labels based on the requirements to train the CLAM SB model. 
Make necessary changes in the code clam_csv_generator.py before running the script.

```
low_expression_genes_folder = "GENE_EXP_CLAM/low_expression_genes"
high_expression_genes_folder = "GENE_EXP_CLAM/high_expression_genes"

output_csv_path = "GENE_EXP_FILE.csv"
```

`python3 clam_csv_generator.py`

The GENE_EXP_FILE.csv should look like below.

```
patient_id, slide_id, label
patient_0, slide_1, low/high_gene_expressions
```

Now, we define the task in create_split_seq.py, main.py, and eval.py, the snippet shown below represents the task, add task based on the requirements, we are doing a binary classification, so we modified the snippet below to reflect our approach, and csv_path should be the file, that we just created in the previous step.

```
if args.task == 'gene_exp':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = GENE_EXP_FILE.csv,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'low_expression_genes':0, 'high_expression_genes':1},
                            patient_strat=True,
                            ignore=[])
```

Finally, add the arguments below to optimize the pipelines in splits, main, and eval.py scripts.

`parser.add_argument('--task', type=str, choices=['gene_exp','task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])`

We will be using this GENE_EXP_CLAM going forward, and you can discard previous folders, if space constraint is an issue. We proceed with creating splits using K-fold cross-validation. 

Navigate to the Classification folder using `cd Classification`

`python3 create_splits_seq.py --task gene_exp --seed 1 --k 12`

--task is the task we just defined, gene_exp

--k is the number of folds, to make splits in the dataset. 10-fold follow distribution of (80/10/10) splits for the dataset. Ours was k=12 or (76/12/12).

To run the training without interruptions run the command shown below for Ubuntu users

`tmux new -s “session”`

You can see active sessions by running,

`tmux ls`

If you already have one, then run

`tmux attach -t “session”`

The split dataset will be saved under the splits folder, and we will use that for training the model. For the training process, run the command below (--embed_dim should be 512 
for CONCH, 1024 for UNI and resnet-50). GENE_EXP_CLAM would be the folder where you save the biomarker-specific low expressions and high expressions 
h5, pt files. Run training and evaluation in tmux. split_dir is the split folder location that it would take for training.

`CUDA_VISIBLE_DEVICES=0 python3 main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 12 --split_dir gene_exp --exp_code gene_exp_100 --weighted_sample --bag_loss ce --task gene_exp --model_type clam_sb --log_data --data_root_dir GENE_EXP_CLAM --embed_dim 1024 >train.txt 2>&1 &`

Here `>train.txt 2>&1 &` saves the progress in train.txt, and assigns a process ID to it, this makes the process uninterruptable and you can run the code smoothly.

Our approach follows training the clam SB model using 95% confidence intervals, and so when the training completes, make sure to save the results folder with corresponding biomarkers, if the process is sequential, with multiple biomarkers.

train.txt is the log file generated after the command is run.

To evaluate the script execute the command below,

`CUDA_VISIBLE_DEVICES=0 python3 eval.py --k 12 --models_exp_code SAVED_MODEL_RESULTS --save_exp_code EVAL_MODEL_RESULTS --task gene_exp --model_type clam_sb --results_dir results 
-data_root_dir GENE_EXP_CLAM --embed_dim 1024>eval.txt 2>&1 &`

The results will be saved in the eval folder, and we will proceed with generating heatmaps now.

eval.txt is the log file generated after the command is run.

Make sure to change the name of the csv file corresponding to the biomarker in the main.py and eval.py scripts.

## Heatmap Generation

To generate Heatmaps for these biomarkers, first navigate to Heatmap/heatmaps/configs/config_template.yaml, then inside it, make changes 
like below. data_dir here would be ORIGINAL_SLIDES_FOLDER, use raw_save_dir to save generated heatmaps. 

The heatmap_demo.csv file within the process_lists folder should look something like below. Make sure to include only the test set images to see the best performance of the model.

```
slide_id, label
slide_1, low/high_gene_expressions
```

```
--- 
exp_arguments:
  # number of classes
  n_classes: 2
  # name tag for saving generated figures and assets
  save_exp_code: HEATMAP_OUTPUT 
  # Where to save raw asset files
  raw_save_dir: heatmaps/HEATMAP_RAW_RESULTS
  # Where to save final heatmaps
  production_save_dir: heatmaps/HEATMAP_PRODUCTION_RESULTS
  # Where to save final heatmaps
  batch_size: 256
data_arguments: 
  # Where is data stored; can be a single str path or a dictionary of key, data_dir mapping
  data_dir: ORIGINAL_SLIDES_FOLDER
  # column name for key in data_dir (if a dictionary mapping is used)
  data_dir_key: None
  # csv list containing slide_ids (can additionally have seg/patch parameters, class labels, etc.)
  process_list: heatmap_demo.csv
  # preset file for segmentation/patching
  preset: presets/tcga.csv
  # file extension for slides
  slide_ext: .svs
  # label dictionary for str: integer mapping (optional)
  label_dict:
    low_expression_genes: 0
    high_expreession_genes: 1                        
patching_arguments:
  # Arguments for patching
  patch_size: 256
  overlap: 0.5
  patch_level: 0
  custom_downsample: 1
encoder_arguments:
  # Arguments for the pretrained encoder model
  model_name:  uni_v1 # currently support: resnet50_trunc, uni_v1, conch_v1
  target_img_size: 224 # Resize images to this size before feeding them to the encoder
model_arguments: 
  # Arguments for initializing the model from checkpoint
  ckpt_path: BEST_MODEL_CHECKPOINTS # split that had the best-performing model checkpoints after training
  model_type: clam_sb # see utils/eval_utils/
  initiate_fn: initiate_model # see utils/eval_utils/
  model_size: small
  drop_out: 0.
  embed_dim: 1024
```

Additionally, make changes to the config file like below.

```
sample_arguments:
  samples:
    - name: "topk_high_attention"
      sample: true
      seed: 1
      k: 10 # save top-k patches
      mode: topk
```

`CUDA_VISIBLE_DEVICES=0 python3 create_heatmaps.py --config config_template.yaml`
