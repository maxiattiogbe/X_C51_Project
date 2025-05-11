import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import Descriptors
from molvs import standardize_smiles


def process_molec(smiles_string):
    smiles_standardized = standardize_smiles(smiles_string)
    mol = Chem.MolFromSmiles(smiles_standardized)
    return mol


def get_molecules(smiles_csv):
    '''
    Read in smiles and get molecules from RDKit
    '''
    df = pd.read_csv(smiles_csv)
    all_smiles = df['smiles'].tolist()
    nprocs = mp.cpu_count()
    with mp.Pool(processes=nprocs) as pool:
        mols = pool.starmap(process_molec, [(i,) for i in all_smiles])
    return all_smiles, mols


def morgan_fps(mols, name, r, bits):
    '''
    Generate Morgan Fingerprints using RDKit
    '''
    mfp = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=bits)
    fps = [mfp.GetFingerprint(mol) for mol in mols]
    mfps = np.asarray(fps)
    np.savez_compressed(f'./{name}/morgan_fps_r{r}_{bits}bits.npz', mfps)
    return mfps


def get_props(mol, props):
    '''
    Calculate properties for a single molecule (mol) using RDKit and extract
    properties (in props)
    '''
    data = Descriptors.CalcMolDescriptors(mol)
    subset = {prop : data[prop] for prop in props}
    return subset


def properties(mols, smiles, name):
    '''
    Calculate properties using RDKit and extract a few physical properties of
    interest
    '''
    props = ['qed', 'SPS', 'MolWt', 'MaxPartialCharge', 'MinPartialCharge',
             'TPSA', 'MolLogP', 'MolMR']

    nprocs = mp.cpu_count()
    with mp.Pool(processes=nprocs) as pool:
        all_data = pool.starmap(get_props, [(mol, props) for mol in mols])

    # Convert to df and add smiles info
    df = pd.DataFrame(all_data)
    df.insert(0, 'smiles', smiles)
    df.to_csv(f'./{name}/physical_properties.csv')


def calculate_all():
    '''
    Calculate physical properties and representations for all sets of molecules:
    - 50k library of molecules from ZINC, Enamine, ChEMBL and BindingDB
    - compounds known to bind from the literatures
    - generated molecules (3 different runs
    '''
    sets = {'library_molecs' : '../smiles/smiles_50k.csv',
            'literature_binders' : '../../previous_mtb_MraY_ligands.csv',
            'gen_mr1_unfiltered_policy' : '../../SyntheMol_generated_candidate_molecules/synthemol_molecules_max_reactions_1_with_policy_unfiltered.csv',
            'gen_mr1_filtered' : '../../SyntheMol_generated_candidate_molecules/synthemol_molecules_max_reactions_1_filtered.csv',
            'gen_mr2_filtered' : '../../SyntheMol_generated_candidate_molecules/synthemol_molecules_max_reactions_2_filtered.csv'}

    for name, csv_file in sets.items():
        # Make directories for results and get molecules
        os.system(f'mkdir {name}')
        smiles, mols = get_molecules(csv_file)

        # Calculate physical properties
        properties(mols, smiles, name)

        # Calculate morgan fingerprints - 3 different parameter pairs
        morgan_fps(mols, name, r=4, bits=1024)
        morgan_fps(mols, name, r=2, bits=1024)
        morgan_fps(mols, name, r=4, bits=2048)

calculate_all()
