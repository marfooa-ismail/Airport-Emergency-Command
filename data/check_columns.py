import pandas as pd

files = [
    "airline_accidents.csv",
    "faa_incidents_data.csv",
    "ntsb_aviation_data.csv",
    "world_aircraft_accident_summary.csv"
]

for file in files:
    df = pd.read_csv(file, nrows=5)
    print("\nFILE:", file)
    print("COLUMNS:")
    print(df.columns.tolist())
    print(df.head())