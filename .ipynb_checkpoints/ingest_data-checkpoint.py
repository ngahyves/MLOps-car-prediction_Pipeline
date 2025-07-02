# ingest_data.py

import pandas as pd
import sqlite3
import os

# --- Configuration ---
# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, 'vehicles.csv')
DB_PATH = os.path.join(BASE_DIR, 'cars.db')
TABLE_NAME = 'car_ads'


def ingest_data_from_csv(csv_path, db_path, table_name):
    """
    Reads data from a CSV file and loads it into a table in a SQLite database.
    
    This function simulates the first step in a data pipeline: raw data ingestion.
    The table is replaced each time the function is executed to ensure a clean starting state.
    """
    # Vérify the existing file
    if not os.path.exists(csv_path):
        print(f"ERROR : File '{csv_path}' not found.")
        print("Download on kaggle and put it in the same folder.")
        return

    try:
        # --- Step 1: read the CSV ---
        print(f"Read : {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"-> {len(df)} rows and {len(df.columns)} columns.")
        print("First rows:")
        print(df.head())
        print("\nColumns infos :")
        df.info()

        # --- Step 2:Connect to the data base ---
        
        conn = sqlite3.connect(db_path)
        
        # Load the Data Frame in  SQLite table.
        print(f"\nData loading '{table_name}' of the data base '{db_path}'...")
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # close the connection
        conn.close()
        
        print("\n--- Data ingestion ! ---")
        print(f"Data base '{db_path}' created/updated.")
        
    except Exception as e:
        print(f"Unexpected error : {e}")

# --- Script entry point ---
if __name__ == '__main__':
    # We launch the main function
    ingest_data_from_csv(CSV_FILE_PATH, DB_PATH, TABLE_NAME)