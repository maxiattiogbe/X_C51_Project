# filtered_bindingdb_mray_inhibitors.tsv
# filtered_chembl_mray_inhibitors.csv
import os
import pandas as pd

bindingdb = pd.read_csv("original_bindingdb_mray_inhibitors.tsv", sep="\t")
chembl = pd.read_csv("original_chembl_mray_inhibitors.csv")

# Keep only specified columns
bindingdb = bindingdb[['Ligand SMILES', 'IC50 (nM)']]
chembl = chembl[['Molecular Weight', 'Smiles', 'AlogP', 'Standard Type', 'Standard Value', 'Standard Units']]

# Remove rows with empty SMILES
bindingdb = bindingdb.dropna(subset=['Ligand SMILES'])
chembl = chembl.dropna(subset=['Smiles'])

# Drop duplicate rows
bindingdb = bindingdb.drop_duplicates()
chembl = chembl.drop_duplicates()

# print("CHEMBL")
# print(chembl.columns)
# print(chembl.shape)
# # # print(chembl.head())
# # print()

# print("BINDINGDB")
# print(bindingdb.columns)
# print(bindingdb.shape)
# # print(bindingdb.head())

bindingdb.to_csv("bindingdb.csv", index=False)
chembl.to_csv("chembl.csv", index=False)

# Extract SMILES strings into separate dataframes
bindingdb_smiles = pd.DataFrame(bindingdb['Ligand SMILES'][:])
chembl_smiles = pd.DataFrame(chembl['Smiles'][:])

bindingdb_smiles = bindingdb_smiles.rename(columns={'Ligand SMILES': 'Smiles'})

print("BINDINGDB SMILES")
# print(bindingdb_smiles.columns)
print(bindingdb_smiles.shape)

print("CHEMBL SMILES")
# print(chembl_smiles.columns)
print(chembl_smiles.shape)

# Remove duplicates from chembl that are in bindingdb
chembl_smiles = chembl_smiles[~chembl_smiles['Smiles'].isin(bindingdb_smiles['Smiles'])]

# Save SMILES-only dataframes
bindingdb_smiles.to_csv("bindingdb_smiles.csv", index=False)
chembl_smiles.to_csv("chembl_smiles.csv", index=False)

# Read all HA files
ha_files = []
for filename in os.listdir('HA'):
    if filename.endswith('.smi'):
        df = pd.read_csv(f'HA/{filename}', sep='\s+')
        df.columns = ['Smiles', 'ID']
        ha_files.append(df)

# Concatenate all dataframes
zinc_smiles = pd.concat(ha_files, ignore_index=True)

# Keep only SMILES column and remove duplicates with chembl/bindingdb
zinc_smiles = zinc_smiles[['Smiles']]
zinc_smiles = zinc_smiles[~zinc_smiles['Smiles'].isin(chembl_smiles['Smiles'])]
zinc_smiles = zinc_smiles[~zinc_smiles['Smiles'].isin(bindingdb_smiles['Smiles'])]

print("ZINC SMILES")
# print(zinc_smiles.columns)
print(zinc_smiles.shape)

zinc_smiles.to_csv("zinc_smiles.csv", index=False)

# Combine all SMILES dataframes
all_smiles = pd.concat([bindingdb_smiles, chembl_smiles, zinc_smiles], ignore_index=True)

print("ALL SMILES")
print(all_smiles.shape)

# Save combined SMILES dataframe
all_smiles.to_csv("all_smiles.csv", index=False)
