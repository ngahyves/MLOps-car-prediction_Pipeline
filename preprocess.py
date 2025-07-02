# preprocess.py

import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime

BASE_DIR= os.path.dirname(os.path.abspath(__file__))
DB_PATH=os.path.join(BASE_DIR, 'cars.db')
INPUT_TABLE_NAME='car_ads'
OUTPUT_CLEANED_CSV=os.path.join(BASE_DIR, 'cleaned_cars.csv')

def load_data_from_db(db_path, table_name):
    print("Start of the preprocessing script")

    #---"Step 1: Loading the data from the data base"---
    print('Loading data from sql lite')
    
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f'Error. The file {DB_PATH} was not found. First run ingest_data.py')
       

        return
    conn = sqlite3.connect(DB_PATH)
    #Using a try/finally block to ensure that the connection is always closed.
    try:
        df=pd.read_sql_query(f'SELECT * FROM {INPUT_TABLE_NAME}', conn)
        print(f"-> {len(df)} Raw data loaded.")
        return df
    finally:
        conn.close()

    

#---Step 2: Data cleaning---

def clean_and_filter_data(df): # Function to clean the data
 
    print ("\n Filtering the data set to remove invalid prices lower than 500$\n")
    df_cleaned =df.copy()
    df_cleaned =df[df['price'] >= 500]

    #Removing duplicates
    print("Removing duplicates") 
    df_cleaned=df_cleaned .drop_duplicates() 

    #Removing missing values
    print("\n Removing missings values\n")
    essential_cols = ['year', 'manufacturer', 'model', 'odometer', 'fuel', 'transmission','title_status', 'price']
    df_cleaned=df_cleaned .dropna(subset=essential_cols)
    
    return df_cleaned 

#---Step 3: Handle outliers

def handle_outliers (df):
    print("\n Step 3: Handle extreme values\n")
    df = df.copy()
    # 3a. Filtering extreme prices with quantiles
    print('\nFiltering extreme prices with quantiles\n')
    lower_price=df['price'].quantile(0.01)
    upper_price=df['price'].quantile(0.99)
    df=df[(df['price']>=lower_price) & (df['price']<=upper_price)]

    #3b.Capping the outliers on other numerical columns
    print('\n Capping outliers\n')
    numeric_cols_to_cap=df.select_dtypes(include='number').drop('price',axis=1, errors='ignore')
    for col in numeric_cols_to_cap:
        Q1= df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR=Q3 -Q1
        lower_bound= Q1 - 1.5 * IQR
        upper_bound=Q3 + 1.5 * IQR
        df[col]=np.clip(df[col], lower_bound, upper_bound)
    return df

# Step 4: Feature engineering and cardinality reduction

# Create the age cars variable
def perform_feature_engineering(df):
    print (' step 4:Feature engineering and cardinality reduction')
    df = df.copy()
    
#4a create the age car column
    df['year']= pd.to_numeric(df['year'], errors='coerce')
    df.dropna(subset=['year'], inplace=True)
    df['year']=df['year'].astype(int)
    current_year = datetime.now().year
    df['age'] = current_year - df['year']
    
#4b  Cardinality reduction for the column 'manufacturer'
    print(" -> Cardinality reduction for the column 'manufacturer'...")
    manufacturer_counts = df['manufacturer'].value_counts()
    top_manufacturers = manufacturer_counts.head(20).index
    df['manufacturer'] = np.where(df['manufacturer'].isin(top_manufacturers), df['manufacturer'], 'other')
    print(f"    'manufacturer' has now {df['manufacturer'].nunique()} categories.")

    # Cardinality reduction for the column 'model'. We will save the TOP 50
    print(" -> Cardinality reduction for the column 'model'...")
    model_counts = df['model'].value_counts()
    top_models = model_counts.head(50).index 
    df['model_reduced'] = np.where(df['model'].isin(top_models), df['model'], 'other')# renaming the column
    df.drop(columns=['model'], inplace=True) # We drop the old column
    print(f"    'model_reduced' has now {df['model_reduced'].nunique()} catégories.")

#4c. Transforming the price variable in logarithm
    df['price_log'] = np.log1p(df['price'])
    
# 4d. Delete constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            df.drop(columns=[col], inplace=True)
            print(f" -> Colonne '{col}' dropped because constant.")
            
    return df

# Step 5: Main function for all the data cleaning process
def get_preprocessed_data():
    print('\nPreprocessing pipeline...')
    df = load_data_from_db(DB_PATH, INPUT_TABLE_NAME)
    df = clean_and_filter_data(df)
    df = handle_outliers(df)
    df = perform_feature_engineering(df)
    
    print("\nSelect final columns...")
    final_columns_to_keep = [
        'price', 'price_log', 'age', 'manufacturer', 
        'model_reduced', 'odometer', 'fuel', 'title_status', 'transmission'
    ]
    existing_columns = [col for col in final_columns_to_keep if col in df.columns]
    df_final = df[existing_columns]
    
    print("-> Simple metadata creations...")
    metadata = { 'categorical_columns': {} }
    categorical_cols_for_dashboard = ['manufacturer', 'model_reduced', 'transmission', 'fuel', 'title_status']
    for col in categorical_cols_for_dashboard:
        metadata['categorical_columns'][col] = sorted(df_final[col].unique().tolist())
    
    print("-> Preprocessing pipeline finished !")
    return df_final, metadata

# Step 6: define the main function
def main():
    df_final = get_preprocessed_data()
    
    # Saving the data set cleaned
    df_final.to_csv(OUTPUT_CLEANED_CSV, index=False)
    print(f"\n Data cleaned in : {OUTPUT_CLEANED_CSV}")
    print(f"the final data set has {len(df_final)} rows and {len(df_final.columns)} columns.")
    print("Final display :")
    print(df_final.head())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Critical error: {e}")
    
