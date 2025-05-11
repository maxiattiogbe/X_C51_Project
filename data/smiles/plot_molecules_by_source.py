import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_molecs_by_source(smiles_sources_csv, smiles_lit_csv, outfile):
    df = pd.read_csv(smiles_sources_csv)
    df_lit = pd.read_csv(smiles_lit_csv)
    libraries = {'enamine_div' : 'Enamine\nDiversity',
                 'enamine_hll' : 'Enamine Hit\nLocator Library',
                 'zinc' : 'ZINC',
                 'chembl' : 'ChEMBL',
                 'bindingdb' : 'BindingDB'}
    libs = list(libraries.keys())
    counts = [df[i].sum() for i in libs] + [len(df_lit)]
    labels = [libraries[i] for i in libs] + ['Literature']
    colors = ['royalblue']*3 + ['lightgreen']*2 + ['tab:green']

    fig, ax = plt.subplots(figsize=[7.5,2.5], dpi=300)
    xs = range(len(libs)+1)
    ax.bar(xs, counts, color=colors, edgecolor='k', lw=0.25)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    for i in range(len(xs)):
        ax.text(xs[i], counts[i]*1.3, f'{counts[i]:,}', ha='center', va='center')
    ax.set_xlabel('Library')
    ax.set_ylabel('Number of Molecules')
    ax.set_yscale('log')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(outfile)


smiles_sources_csv = 'smiles_50k_with_sources.csv'
smiles_lit_csv = '../../previous_mtb_MraY_ligands.csv'
outfile = 'molecs_by_source.png'
plot_molecs_by_source(smiles_sources_csv, smiles_lit_csv, outfile)
