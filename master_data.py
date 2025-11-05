import pandas as pd
import numpy as np

# --- FILE PATHS  ---
file_path_1_xlsx = r'data\raw\NEET_2024_CenterWise_Stats.xlsx'
file_path_2_xlsx = r'data\raw\NEET_2024_Marks_By_State_City_Center.xlsx'
output_file_csv = r'data\NEET_Master_Analysis_Data.csv'


# ----------------- PART 1: CONSOLIDATING THE MARKS DATA (GRANULAR) -----------------

try:
    # 1. Load the Excel file object to read all sheets
    xls = pd.ExcelFile(file_path_2_xlsx)

    # 2. Read all sheets and save them into a list
    df_marks_list = []
    print("\nStarting to load and combine all student score sheets...")
    for sheet_name in xls.sheet_names:
        print(f"  - Loading sheet: {sheet_name}")
        df_marks_list.append(xls.parse(sheet_name))

    # 3. Concatenate (stack) all the sheets into one massive DataFrame
    df_marks = pd.concat(df_marks_list, ignore_index=True)
    print(f"\nTotal student records consolidated: {len(df_marks):,} rows.")

    # 4. Clean the data (remove the unnecessary 'sno' column)
    df_marks.drop(columns=['sno'], inplace=True)

except FileNotFoundError:
    print(f"\nERROR: File not found at: {file_path_2_xlsx}")
    print("Please ensure the Excel file is in the correct folder.")
    exit()

# ----------------- PART 2: AGGREGATING ADVANCED FEATURES (ML FEATURES) -----------------

print("\n--- Aggregating Advanced Statistical Features for ML (Corrected) ---")

# Group by 'center_id' and calculate advanced statistics
df_marks_agg = df_marks.groupby('center_id')['marks'].agg(
    Center_Std_Dev = 'std',
    Center_Skewness = pd.Series.skew,  # FIX: Using the callable function
    Center_Kurtosis = pd.Series.kurt   # FIX: Using the callable function
).reset_index()

# Round the results for better readability
df_marks_agg[['Center_Std_Dev', 'Center_Skewness', 'Center_Kurtosis']] = (
    df_marks_agg[['Center_Std_Dev', 'Center_Skewness', 'Center_Kurtosis']].round(3)
)

print("First 3 rows of the new ML features table:")
print(df_marks_agg.head(3))

# ----------------- PART 3: FINAL MASTER DATA MERGE -----------------

print("\n--- Merging DataFrames and Finalizing Master File ---")

# 1. Load the CenterWiseStats file
df_center_stats = pd.read_excel(file_path_1_xlsx)

# 2. Re-create the key inequality metrics
df_center_stats['Ultra_High_Score_Ratio'] = (df_center_stats['above_700_marks'] / df_center_stats['total_students'])
df_center_stats['High_Score_Ratio'] = ((df_center_stats['above_600_marks'] + df_center_stats['above_700_marks']) / df_center_stats['total_students'])
df_center_stats['Center_v_National_Gap'] = (df_center_stats['center_average_marks'] - df_center_stats['national_average_marks'])
df_center_stats['Center_v_State_Gap'] = (df_center_stats['center_average_marks'] - df_center_stats['state_average_marks'])

# 3. Merge the aggregated ML features (df_marks_agg) into the center stats table
df_master = df_center_stats.merge(
    df_marks_agg,
    on='center_id',
    how='left'
)

# 4. Final check and save
print("\n--- Final Master DataFrame Check ---")
print(f"Total columns in Master File: {len(df_master.columns)}. (Should be 18 columns)")
print(df_master[['center_name', 'Center_Skewness', 'Center_Kurtosis']].sample(3))

# Save the final Master DataFrame to a new CSV file
df_master.to_csv(output_file_csv, index=False)
print(f"\n✅ PHASE 1 COMPLETE! Your single, powerful Master Data File is saved as {output_file_csv}.")