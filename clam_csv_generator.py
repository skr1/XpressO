import os
import pandas as pd

# Define the folders
low_expression_genes_folder = "/home/ubuntu/mmp3_191_gene_exp_clam_UNI/low_expression_genes"
high_expression_genes_folder = "/home/ubuntu/mmp3_191_gene_exp_clam_UNI/high_expression_genes"

# Subfolders containing h5 files
low_h5_folder = os.path.join(low_expression_genes_folder, "h5_files")
high_h5_folder = os.path.join(high_expression_genes_folder, "h5_files")

# List to store data
data = []

# Set to track processed files to avoid duplicates
processed_files = set()

# Dictionary to store case IDs
case_id_map = {}

# Helper function to parse files and append data
def parse_files(folder, label):
    files = os.listdir(folder)
    for filename in files:
        if filename.endswith(".h5") and filename not in processed_files:
            processed_files.add(filename)
            slide_id = os.path.splitext(filename)[0]  # Using the filename without extension as slide_id
            if slide_id not in case_id_map:
                case_id_map[slide_id] = f"patient_{len(case_id_map)}"  # Assign a unique patient ID
            case_id = case_id_map[slide_id]
            data.append([case_id, slide_id, label])

# Parse low expression folder
parse_files(low_h5_folder, "low_expression_genes")

# Parse high expression folder
parse_files(high_h5_folder, "high_expression_genes")

# Create a DataFrame
df = pd.DataFrame(data, columns=["case_id", "slide_id", "label"])

# Save to CSV
output_csv_path = "/home/ubuntu/mmp3_191_gene_expression.csv"
df.to_csv(output_csv_path, index=False)

print("CSV file has been created successfully.")
