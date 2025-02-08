import pandas as pd
import os
import glob
import shutil
import argparse

def main(args):
    # Load the DataFrame
    ald_gene = pd.read_csv(args.csv_file, delimiter=',', header = 0)
    column_name = args.column_name

    # Check if column exists
    if column_name not in ald_gene.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Compute the median of the column
    median_value = ald_gene[column_name].median()

    # Binarize the column based on the median
    ald_gene['binarized'] = ald_gene[column_name].apply(lambda x: 1 if x > median_value else 0)

    # Create subfolders if they don't exist
    os.makedirs(args.low_h5_folder, exist_ok=True)
    os.makedirs(args.low_pt_folder, exist_ok=True)
    os.makedirs(args.high_h5_folder, exist_ok=True)
    os.makedirs(args.high_pt_folder, exist_ok=True)

    # Get all files in the source folders
    all_h5_files = glob.glob(os.path.join(args.h5_source_folder, "*.h5"))
    all_pt_files = glob.glob(os.path.join(args.pt_source_folder, "*.pt"))

    # Combine all files into a single list
    all_files = all_h5_files + all_pt_files

    # Move files to respective folders based on 'binarized' values
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Extract gene ID from filename
        gene_id_with_suffix = filename.split("_")[0]  # Extract gene ID with suffix
        
        # Remove suffix to get the correct gene ID format
        gene_id_parts = gene_id_with_suffix.split(".")
        if len(gene_id_parts) > 1:
            gene_id = gene_id_parts[0]  # Extract only "TCGA-XX-XXXX" format
        else:
            gene_id = gene_id_with_suffix
        
        # Remove additional suffix "-01Z-00-DX1" if present
        gene_id = gene_id.split("-01Z-00-DX1")[0]
        
        # Check if the gene ID is in the DataFrame
        if gene_id in ald_gene['ID'].values:
            # Get the binarized value for this gene ID
            binarized_value = ald_gene.loc[ald_gene['ID'] == gene_id, 'binarized'].values[0]
            
            if binarized_value == 0:
                # Move to low expression folder
                if filename.endswith(".h5"):
                    destination = os.path.join(args.low_h5_folder, filename)
                elif filename.endswith(".pt"):
                    destination = os.path.join(args.low_pt_folder, filename)
                shutil.copy(file_path, destination)
                print(f"Moved {filename} to low expression folder.")
            elif binarized_value == 1:
                # Move to high expression folder
                if filename.endswith(".h5"):
                    destination = os.path.join(args.high_h5_folder, filename)
                elif filename.endswith(".pt"):
                    destination = os.path.join(args.high_pt_folder, filename)
                shutil.copy(file_path, destination)
                print(f"Moved {filename} to high expression folder.")
        else:
            print(f"Gene ID {gene_id} not found in DataFrame.")

    print("Files moved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binarize gene expression data and sort files accordingly.')
    
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing gene expression data.')
    parser.add_argument('--column_name', type=str, required=True, help='Name of the column to binarize.')
    parser.add_argument('--h5_source_folder', type=str, required=True, help='Source folder containing .h5 files.')
    parser.add_argument('--pt_source_folder', type=str, required=True, help='Source folder containing .pt files.')
    parser.add_argument('--low_h5_folder', type=str, required=True, help='Destination folder for low expression .h5 files.')
    parser.add_argument('--low_pt_folder', type=str, required=True, help='Destination folder for low expression .pt files.')
    parser.add_argument('--high_h5_folder', type=str, required=True, help='Destination folder for high expression .h5 files.')
    parser.add_argument('--high_pt_folder', type=str, required=True, help='Destination folder for high expression .pt files.')

    args = parser.parse_args()
    main(args)