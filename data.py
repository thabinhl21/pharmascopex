import pandas as pd

df = pd.read_csv('GDSC1and2_w_CellLineData.csv')

unique_drugs = df['DRUG_NAME'].unique().tolist()

# print(unique_drugs)

unique_cell = df['CELL_LINE_NAME'].unique().tolist()

print(unique_cell)

