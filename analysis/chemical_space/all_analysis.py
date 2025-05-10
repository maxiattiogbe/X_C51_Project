import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import Draw


def get_all_data(smiles_sources_csv, docking_scores_csv, phys_prop_csv, fp_path):
    '''
    Get SMILES strings for the molecules, information about sources, and
    combine with docking scores, physical properties, and Morgan fingerprints.
    '''
    df_smiles = pd.read_csv(smiles_sources_csv)
    df_dock = pd.read_csv(docking_scores_csv)
    df_dock.columns = ['smiles', 'TotalEnergy']
    df_smiles.drop(['Unnamed: 0'], axis=1, inplace=True)
    df = pd.merge(df_smiles, df_dock, how='left', on='smiles')

    # Which molecules do we have docking scores for?
    all_smiles = df_smiles['smiles'].tolist()
    docked_smiles = df_dock['smiles'].tolist()
    indices_to_keep = [i for i in range(len(all_smiles)) if all_smiles[i] in docked_smiles]
    df_subset = df.loc[indices_to_keep]

    # Next, load data on physical properties
    props_df = pd.read_csv(phys_prop_csv)
    props = [i for i in props_df.columns if i not in ['smiles', 'Unnamed: 0']]
    titles = {'qed' : 'Quantitative Estimate\nof Druglikeness',
              'SPS' : 'Spatial Complexity\nScore',
              'MolWt' : 'Molecular Weight',
              'MaxPartialCharge' : 'Maximum Partial Charge',
              'MinPartialCharge' : 'Minimum Partial Charge',
              'TPSA' : 'Total Polar Surface Area',
              'MolLogP' : 'LogP',
              'MolMR' : 'Molar Refractivity'}
    # only keep molecules for which we already have docking scores
    props_df_subset = props_df.loc[indices_to_keep]

    # Next, load Moran fingerprints
    fps = np.load(fp_path)['arr_0']
    # only keep fingerprints for molecules for which we have docking scores
    fps_subset = fps[indices_to_keep,:]

    return df_subset, props_df_subset, props, titles, fps_subset


def plot_docking_score_distr_comparisons(df, outfile):
    '''
    Plot distributions of docking scores (left) and docking score by library (right).
    Probably binders for MraY (from ChEMBL and BindingDB) are expected to have lower docking scores
    than those from the Enamine and Zinc libraries
    '''
    df_good = df[(df['chembl'] == True) | (df['bindingdb'] == True)]
    df_other = df[(df['zinc'] == True) | (df['enamine_div'] == True) | (df['enamine_hll'] == True)]
    all_scores = df['TotalEnergy'].values
    good_molec_scores = df_good['TotalEnergy'].values
    other_molec_scores = df_other['TotalEnergy'].values

    bins = np.arange(-180, 0, 10)

    fig, axs = plt.subplots(1, 2, figsize=[7.5, 2.5], dpi=300)
    axs[0].hist(all_scores, bins=bins, label='All Molecules',
                color='lightblue', edgecolor='k', lw=0.25)
    axs[0].set_yscale('log')
    axs[0].legend(fontsize=8)

    axs[1].hist(good_molec_scores, bins=bins, label='Molecules from ChEMBL\nand BindingDB\n(probable binders)',
                color='tab:green', edgecolor='k', lw=0.25, alpha=1, zorder=10)
    axs[1].hist(other_molec_scores, bins=bins, label='Molecules from Enamine\nand Zinc Libraries',
                color='tab:gray', edgecolor='k', lw=0.25, alpha=0.5)
    axs[1].set_yscale('log')
    axs[1].legend(fontsize=6)

    for i in range(2):
        axs[i].set_xlabel(r'Docking score / kcal mol$^{-1}$')
        axs[i].set_ylabel('Number of Molecules')
        axs[i].set_xlim(-180, 0)
        axs[i].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    
    fig.savefig(outfile)


def identify_best_binders(df):
    '''
    Identify the best binders in the dataset from docking scores
    '''
    df_good = df[(df['chembl'] == True) | (df['bindingdb'] == True)]
    df_other = df[(df['zinc'] == True) | (df['enamine_div'] == True) | (df['enamine_hll'] == True)]
    best_scores = np.sort(df['TotalEnergy'].values)[0:10]

    df_best = df[df['TotalEnergy'] < -169]
    best_enamine = df_best[df_best['enamine_div']]['smiles'].tolist()
    best_zinc = df_best[df_best['zinc'] == True]['smiles'].tolist()
    best_chembl = df_best[df_best['chembl'] == True]['smiles'].tolist()
    best_bindingdb = df_best[df_best['bindingdb'] == True]['smiles'].tolist()

    return best_enamine, best_zinc, best_chembl, best_bindingdb


def plot_best_binders(best_enamine, best_zinc, best_chembl, best_bindingdb):
    '''
    Make figures of best binders using rdkit
    Separate known binders (chembl, bindingdb) from molecules from Enamine and ZINC libraries
    '''
    mols1 = [Chem.MolFromSmiles(i) for i in best_enamine + best_zinc]
    im_best_enamine_zinc = Draw.MolsToGridImage(mols1, molsPerRow=3, subImgSize=[600, 300], returnPNG=False)
    im_best_enamine_zinc.save('best_binders_enamine_zinc.png')

    mols2 = [Chem.MolFromSmiles(i) for i in best_chembl + best_bindingdb]
    im_best_known = Draw.MolsToGridImage(mols2, molsPerRow=2, subImgSize=[600, 300], returnPNG=False)
    im_best_known.save('best_binders_chembl_bindingdb.png')


def plot_distrs_of_properties(props_df, props, titles, outfile):
    '''
    Plot distributions of physical properties of the molecules
    '''
    fig, axes = plt.subplots(2, 4, dpi=300, figsize=[7.5, 3])
    axs = axes.ravel()
    for i in range(len(props)):
        ax = axs[i]
        prop = props[i]
        ax.hist(props_df[prop].values, bins=25, color='lightblue',
                edgecolor='k', lw=0.25)
        ax.set_xlabel(titles[prop], fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(axis='both', labelsize=6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig(outfile)


def plot_distrs_of_properties_2d(df, props_df, props, titles, outfile):
    '''
    Plot distributions of physical properties of the molecules - all by all
    '''
    nprops = len(props)
    fs = 6
    fig, axs = plt.subplots(nprops, nprops, dpi=300, figsize=[12, 12])
    for i in range(nprops):
        for j in range(nprops):
            ax = axs[i,j]
            prop1 = props[i]
            prop2 = props[j]
            if i == j:
                ax.hist(props_df[prop1].values, bins=25, color='tab:gray',
                        edgecolor='k', lw=0.25)
                ax.set_yscale('log')
                ax.set_xlabel(titles[prop1], fontsize=fs)
                ax.set_ylabel('Frequency', fontsize=fs)
            else:
                ax.scatter(props_df[prop1], props_df[prop2], s=2,
                           c=df['TotalEnergy'], cmap='magma',
                           vmin=-180, vmax=-75)
                ax.set_xlabel(titles[prop1], fontsize=fs)
                ax.set_ylabel(titles[prop2], fontsize=fs)

            ax.tick_params(axis='both', labelsize=6)

    plt.tight_layout()
    fig.savefig(outfile)


def center_standardize(data):
    data_clean = (data - np.mean(data, axis=0)) / np.nanstd(data, axis=0)
    return data_clean


def calc_pca(ncomps, data):
    '''
    pca_proj: projection of data onto principal components
    pca_exp_var: explained variance ratio of each PC
    '''
    pca = PCA(ncomps)
    pca_proj = pca.fit_transform(data)
    pca_exp_var = pca.explained_variance_ratio_
    return pca_proj, pca_exp_var


def plot_pca_all(pca_proj, pca_exp_var, df, title, outfile):
    fig, axs = plt.subplots(1, 2, figsize=[7.5,3.25], dpi=300)
    pc1_var = pca_exp_var[0]
    pc2_var = pca_exp_var[1]
    df = df.reset_index()

    # Colour by library
    libraries = {'zinc' : 'ZINC',
                 'enamine_div' : 'Enamine\nDiversity',
                 'enamine_hll' : 'Enamine Hit\nLocator Library',
                 'chembl' : 'ChEMBL',
                 'bindingdb' : 'BindingDB'}
    colors = {'zinc' : 'tab:gray',
              'enamine_div' : 'tab:green',
              'enamine_hll' : 'tab:blue',
              'chembl' : 'tab:red',
              'bindingdb' : 'tab:orange'}
    for lib in list(libraries.keys()):
        indices = df.index[df[lib] == True].to_numpy()
        axs[0].scatter(pca_proj[indices,0], pca_proj[indices,1],
                       label=libraries[lib], color=colors[lib],
                       alpha=0.9, s=2.5, lw=0)
    axs[0].set_xlabel(f'PC 1 ({100*pc1_var:.2f} % of Variance)')
    axs[0].set_ylabel(f'PC 1 ({100*pc2_var:.2f} % of Variance)')
    axs[0].set_title('Molecules in Dataset,\nColored by Library', fontsize=10)
    axs[0].legend(fontsize=8, labelspacing=0.25)

    # Colour by docking scores
    docking_scores = df['TotalEnergy'].values
    im = axs[1].scatter(pca_proj[:,0], pca_proj[:,1], s=2.5, lw=0,
                        c=df['TotalEnergy'], cmap='magma',
                        vmin=np.min(docking_scores), vmax=-75)
    axs[1].set_xlabel(f'PC 1 ({100*pc1_var:.2f} % of Variance)')
    axs[1].set_ylabel(f'PC 1 ({100*pc2_var:.2f} % of Variance)')
    axs[1].set_title('Molecules in Dataset,\nColored by Docking Score', fontsize=10)
    fig.colorbar(im, label=r'Docking Score / kcal mol$^{-1}$')

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(outfile)


# Load all data
smiles_sources_csv = '../../data/smiles/smiles_50k_with_sources.csv'
docking_scores_csv = '../../min_energy.csv'
phys_prop_csv = '../../data/representations/physical_properties.csv'
fp_path = '../../data/representations/morgan_fps_r4_1024bits.npz'
df, props_df, props, titles, fps = get_all_data(smiles_sources_csv, docking_scores_csv, phys_prop_csv, fp_path)

# Plot distributions of docking scores, separated by library
plot_docking_score_distr_comparisons(df, outfile='docking_score_distrs_w_comparison.png')

# Identify best binders and make figures
best_enamine, best_zinc, best_chembl, best_bindingdb = identify_best_binders(df)
plot_best_binders(best_enamine, best_zinc, best_chembl, best_bindingdb)

# Plot distributions of physical properties
plot_distrs_of_properties(props_df, props, titles, outfile='distr_phys_prop_1d_hist.png')
plot_distrs_of_properties_2d(df, props_df, props, titles, outfile='distr_phys_prop_2d_hist.png')

'''
# Calculate PCA on physical property vectors and plot results
data_props_all = np.array(props_df.values[:,2:]).astype(np.float32)
data_props = center_standardize(data_props_all)
props_pca_proj, props_pca_exp_var = calc_pca(8, data_props)
plot_pca_all(props_pca_proj, props_pca_exp_var, df,
             title='PCA on Physical Property Vectors',
             outfile='pca_phys_prop.png')

# Calculate PCA on Morgan fingerprints and plots results
data_fps_all = np.array(fps)
data_fps = center_standardize(data_fps_all)
fps_pca_proj, fps_pca_exp_var = calc_pca(100, data_fps)
plot_pca_all(fps_pca_proj, fps_pca_exp_var, df,
             title='PCA on Morgan Fingerprints',
             outfile='pca_morgan_fps.png')
'''
