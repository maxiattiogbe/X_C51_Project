import pandas as pd

# Read in lists of smiles for molecules from different sources
dirpath = './molecules_by_source'
df_chembl = pd.read_csv(f'{dirpath}/chembl_smiles.csv')
df_zinc = pd.read_csv(f'{dirpath}/zinc_smiles.csv')
df_bindingdb = pd.read_csv(f'{dirpath}/bindingdb_smiles.csv')
df_enamine_div = pd.read_csv(f'{dirpath}/enamine_div.csv')
df_enamine_hll = pd.read_csv(f'{dirpath}/enamine_hll.csv')

# Gather smiles information
enamine_hll = df_enamine_hll['SMILES'].tolist()
enamine_div = df_enamine_div['SMILES'].tolist()
bindingdb = df_bindingdb['Smiles'].tolist()
zinc = df_zinc['Smiles'].tolist()
chembl = df_chembl['Smiles'].tolist()

# Remove repeated molecules
full_set_smiles = sorted(list(set(enamine_hll + enamine_div + bindingdb + zinc + chembl)))
print(len(full_set_smiles))

# Create a new dataframe for set of molecules
df = pd.DataFrame(columns=['smiles', 'enamine_div', 'enamine_hll', 'chembl', 'bindingdb', 'zinc'])

# Keep track of sources of molecules
df['smiles'] = full_set_smiles
df['enamine_hll'] = [smiles in enamine_hll for smiles in full_set_smiles]
df['enamine_div'] = [smiles in enamine_div for smiles in full_set_smiles]
df['bindingdb'] = [smiles in bindingdb for smiles in full_set_smiles]
df['chembl'] = [smiles in chembl for smiles in full_set_smiles]
df['zinc'] = [smiles in zinc for smiles in full_set_smiles]

df.to_csv('full_set_smiles_with_sources.csv')
