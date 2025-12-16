import pandas as pd
import numpy as np

def merge_patient_info(extracted_file, marked_file, output_file):
    """
    Merge patient information from two CSV files. 
    
    For each patient (matched by ID in first column), prefer non-missing values 
    from extracted_file, then fall back to marked_file values. 
    Missing values are represented as -1.
    
    Args:
        extracted_file: Path to primary data source CSV
        marked_file: Path to secondary data source CSV
        output_file: Path to output merged CSV
    """
    df_extracted = pd.read_csv(extracted_file, na_values=['n/a', 'N/A', 'na', 'NA', '', ' '], keep_default_na=True)
    df_marked = pd.read_csv(marked_file, na_values=['n/a', 'N/A', 'na', 'NA', '', ' '], keep_default_na=True)
    
    id_col = df_extracted.columns[0]
    
    for col in df_extracted.columns:
        if col != id_col:
            df_extracted[col] = pd.to_numeric(df_extracted[col], errors='coerce')
            df_marked[col] = pd.to_numeric(df_marked[col], errors='coerce')
            
            df_extracted[col] = df_extracted[col].replace(-1, np.nan)
            df_marked[col] = df_marked[col].replace(-1, np.nan)
    
    df_merged = df_extracted.merge(
        df_marked, 
        on=id_col, 
        how='inner', 
        suffixes=('_extracted', '_marked')
    )
    
    df_output = df_merged[[id_col]].copy()
    
    numeric_cols = [col for col in df_extracted. columns if col != id_col]
    
    if 'hospital_patient_id_extracted' in df_merged.columns:
        hospital_id = df_merged['hospital_patient_id_extracted']. combine_first(df_merged['hospital_patient_id_marked'])
        df_output['hospital_patient_id'] = hospital_id. fillna(-1).astype(int)
        numeric_cols.remove('hospital_patient_id')
    
    for col in numeric_cols:
        extracted_col = f'{col}_extracted'
        marked_col = f'{col}_marked'
        
        if extracted_col in df_merged. columns and marked_col in df_merged.columns:
            df_output[col] = df_merged[extracted_col].combine_first(df_merged[marked_col])
            df_output[col] = df_output[col].fillna(-1).astype(int)
    
    df_output.to_csv(output_file, index=False)
    
    print(f"Merged {len(df_output)} patients from {extracted_file} and {marked_file}")
    print(f"Output written to {output_file}")


def main():
    lab = False
    mirror_id = 2
    
    if lab:
        extracted_file = 'lab_overall_patient_info.csv'
        marked_file = 'lab_overall_patient_info. csv'
        output_file = 'lab_merged_patient_info.csv'
    else:
        extracted_file = f'overall_patient_info_{mirror_id}.csv'
        marked_file = f'extracted_vitals_{mirror_id}.csv'
        output_file = f'merged_patient_info_{mirror_id}.csv'
    
    merge_patient_info(extracted_file, marked_file, output_file)


if __name__ == '__main__': 
    main()