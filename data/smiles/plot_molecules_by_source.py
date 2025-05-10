import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_molecs_by_source(smiles_sources_csv, outfile):
    df = pd.read_csv(smiles_sources_csv)
    libraries = {'enamine_div' : 'Enamine\nDiversity',
                 'enamine_hll' : 'Enamine Hit\nLocator Library',
                 'zinc' : 'ZINC',
                 'chembl' : 'ChEMBL',
                 'bindingdb' : 'BindingDB'}
    libs = list(libraries.keys())
    counts = [df[i].sum() for i in libs]

    fig, ax = plt.subplots(figsize=[5,2.5], dpi=300)
    xs = range(len(libs))
    ax.bar(xs, counts, color='navy', edgecolor='k', lw=0.25)
    ax.set_xticks(xs)
    ax.set_xticklabels([libraries[i] for i in libs], fontsize=8)
    ax.set_xlabel('Library')
    ax.set_ylabel('Number of Molecules')
    ax.set_yscale('log')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(outfile)


smiles_sources_csv = 'smiles_50k_with_sources.csv'
outfile = 'molecs_by_source.png'
plot_molecs_by_source(smiles_sources_csv, outfile)
