import pandas as pd
import numpy as np
np.random.seed(0)

# Budget
budget = 50000

# Read in smiles
df = pd.read_csv('./full_set_smiles_with_sources.csv')

# Select all molecules from Enamine libraries (15,335 molecules),
# as well as those in chembl (22) and bindingdb (39 molecules)
smiles_to_keep = df.loc[(df['enamine_div'] | df['enamine_hll'] |
                         df['chembl'] | df['bindingdb'])]
num_keep = len(smiles_to_keep)

# Need to subsample the remaining molecules from ZINC
num_to_sample = budget - num_keep

# Count molecules in ZINC that aren't in the other libraries
zinc_unique = df[~df.isin(smiles_to_keep).all(axis=1)].reset_index()
num_zinc_unique = len(zinc_unique)

print(f"Kept {num_keep} molecules from enamine, chembl and bindingdb.")
print(f"Sampling {num_to_sample} molecules from the {num_zinc_unique} molecules in ZINC.")

# Subsample from ZINC
zinc_indices = zinc_unique.index.to_numpy()
zinc_indices_subsampled = np.random.choice(zinc_indices, num_to_sample, replace=False)

# Save indices
np.savetxt("./zinc_indices_subsampled.txt", zinc_indices_subsampled)

# Extract molecules corresponding to subsampled indices and add to smiles_to_keep df
zinc_molecs = zinc_unique.iloc[zinc_indices_subsampled]
all_smiles_to_keep = pd.concat([smiles_to_keep, zinc_molecs], ignore_index=True)

# Clean up
all_smiles_to_keep.drop(['Unnamed: 0', 'index'], axis=1, inplace=True)

# Check that we have 50,000 unique molecules
assert len(list(set(all_smiles_to_keep['smiles'].unique()))) == 50000

# Save to file
all_smiles_to_keep.to_csv('./smiles_50k_with_sources.csv')
all_smiles_to_keep.to_csv('./smiles_50k.csv', columns=['smiles'])
