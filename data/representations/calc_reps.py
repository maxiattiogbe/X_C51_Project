import numpy as np
import pandas as pd
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import Descriptors
from molvs import standardize_smiles


def get_molecules(smiles_csv):
    '''
    Read in smiles and get molecules from RDKit
    '''
    df = pd.read_csv(smiles_csv)
    smiles = df['smiles'].tolist()
    smiles_standardized = [standardize_smiles(i) for i in smiles]
    mols = [Chem.MolFromSmiles(i) for i in smiles_standardized]
    return smiles, mols


def morgan_fps(mols, r, bits):
    '''
    Generate Morgan Fingerprints using RDKit
    '''
    mfp = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=bits)
    fps = [mfp.GetFingerprint(mol) for mol in mols]
    mfps = np.asarray(fps)
    np.savez_compressed(f'./morgan_fps_r{r}_{bits}bits.npz', mfps)
    return mfps


def properties(mols, smiles):
    '''
    Calculate properties using RDKit and extract a few physical properties of
    interest
    '''
    props = ['qed', 'SPS', 'MolWt', 'MaxPartialCharge', 'MinPartialCharge',
             'TPSA', 'MolLogP', 'MolMR']
    all_data = []
    for i in range(len(mols)):
        mol = mols[i]
        data = Descriptors.CalcMolDescriptors(mol)
        subset = {prop : data[prop] for prop in props}
        all_data.append(subset)

    # Convert to df and add smiles info
    df = pd.DataFrame(all_data)
    df.insert(0, 'smiles', smiles)
    df.to_csv('./physical_properties.csv')


smiles_csv = '../smiles/smiles_50k.csv'
smiles, mols = get_molecules(smiles_csv)
properties(mols, smiles)
morgan_fps(mols, r=4, bits=1024)
morgan_fps(mols, r=2, bits=512)
