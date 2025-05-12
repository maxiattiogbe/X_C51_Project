import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import Draw
from umap import UMAP
from scipy import stats


def get_smiles_sources_scores(groups, smiles_path, gen_path, dock_path):
    '''
    Get SMILES and sources for all molecules, and process docking scores.
    '''
    # Five sets of molecules with smiles in different files
    smile_files = {'library_molecs' : f'{smiles_path}/smiles_50k_with_sources.csv',
                   'gen_mr1_filtered' : f'{gen_path}/synthemol_molecules_max_reactions_1_filtered.csv',
                   'gen_mr1_unfiltered_policy' : f'{gen_path}/synthemol_molecules_max_reactions_1_with_policy_unfiltered.csv',
                   'gen_mr2_filtered' : f'{gen_path}/synthemol_molecules_max_reactions_2_filtered.csv',
                   'literature_binders' : '../../previous_mtb_MraY_ligands.csv'}
    all_smiles = []
    all_sources = []
    for group in groups:
        df = pd.read_csv(smile_files[group])
        smiles = df['smiles']
        all_smiles.extend(smiles)
        if group == 'library_molecs':
            # For the initial library file, source information is by column (bool)
            df_just_sources = df[['enamine_div', 'enamine_hll', 'chembl', 'zinc', 'bindingdb']]
            sources = df_just_sources.idxmax(axis=1)
        else:
            sources = [group]*len(df)
        all_sources.extend(sources)

    # Combine smiles and sources into dataframe
    molecules = pd.DataFrame({'smiles' : all_smiles,
                              'source' : all_sources})

    # Now, get docking scores: 3 files with docking scores cover the 5 sets, so just
    # read in results, find minimum score, and merge onto existing df.  Not all molecules
    # in the original library were successfully docked, so add nan's instead
    dock_files = [f'{dock_path}/docking_results.csv',
                  f'{dock_path}/SMILE_fitness_predicted_Top50.csv',
                  f'{dock_path}/SMILE_fitness_predicted_Top50_known_Maxi.csv']
    dock_data = []
    for dock_file in dock_files:
        df = pd.read_csv(dock_file)
        dock_data.append(df[['#Ligand', 'Compound', 'SMILE', 'TotalEnergy']])
    all_dock_data = pd.concat(dock_data)

    # Group by smiles string and get minimum energy (ie best docking score)
    docking_grouped = all_dock_data.groupby('SMILE', as_index=False)['TotalEnergy'].min()
    docking_grouped = docking_grouped.rename(columns={'SMILE' : 'smiles'})

    # Merge docking data onto SMILES dataframe
    df_new = pd.merge(molecules, docking_grouped, how='outer', on='smiles')

    # Check that order of molecules has not changed - this means that we can still
    # work with the props_df's and Morgan fingerprints in the order they were written
    # to the files when they were generated (instead of needing to figure out a map)
    old = molecules['smiles']
    new = df_new['smiles']
    assert (old == new).all()

    return df_new


def get_representations(groups, reps_path):
    '''
    Get representations for the molecules (physical properties and Morgan fingerprints)
    and combine into df and dict
    '''
    # Get individual data
    all_phys_prop = []
    mfps_lists = {'r2_1024bits' : [], 'r4_1024bits' : [], 'r4_2048bits' : []}
    for i in range(len(groups)):
        group = groups[i]

        # Physical properties
        df = pd.read_csv(f'{reps_path}/{group}/physical_properties.csv')
        all_phys_prop.append(df)

        # Morgan fingerprints
        for params in mfps_lists.keys():
            mfp_arr = np.load(f'{reps_path}/{group}/morgan_fps_{params}.npz')['arr_0']
            mfps_lists[params].append(mfp_arr)

    # Combine
    df = pd.concat(all_phys_prop, ignore_index=True)
    mfps = {}
    for params in mfps_lists.keys():
        data = np.vstack(mfps_lists[params])
        mfps.update({params : data})

    return df, mfps


def plot_docking_score_distr_comparisons(df, outfile):
    '''
    Plot distributions of docking scores (left) and docking score by library (right).
    Known binders of Mtb MraY from the literature, and probable binders for MraY
    (from ChEMBL and BindingDB) are expected to have lower docking scores
    than those from the Enamine and Zinc libraries.  If the models are performing well,
    the generated molecules should also have low docking scores
    '''
    df_generated = df[df['source'].isin(['gen_mr1_filtered',
                                         'gen_mr1_unfiltered_policy',
                                         'gen_mr2_filtered'])]
    df_probable = df[df['source'].isin(['chembl', 'bindingdb'])]
    df_known = df[df['source'] == 'literature_binders']
    df_libs = df[df['source'].isin(['enamine_div', 'enamine_hll', 'zinc'])]

    all_scores = df['TotalEnergy'].values
    known_scores = df_known['TotalEnergy'].values
    generated_scores = df_generated['TotalEnergy'].values
    probable_scores = df_probable['TotalEnergy'].values
    lib_scores = df_libs['TotalEnergy'].values

    bins = np.arange(-180, -50, 10)

    fig, axs = plt.subplots(1, 2, figsize=[7.5, 2.5], dpi=300)
    axs[0].hist(all_scores, bins=bins, label='All Molecules',
                color='lightblue', edgecolor='k', lw=0.25)
    axs[0].set_yscale('log')
    axs[0].legend(fontsize=7)

    axs[1].hist(generated_scores, bins=bins, label='Generated Molecules',
                color='tab:pink', edgecolor='k', lw=0.25, alpha=0.5, zorder=4)
    axs[1].hist(known_scores, bins=bins, label='\nLiterature\n(known binders)',
                color='tab:green', edgecolor='k', lw=0.25, alpha=0.75, zorder=6)
    axs[1].hist(probable_scores, bins=bins, label='ChEMBL and BindingDB\n(probable binders)',
                color='tab:blue', edgecolor='k', lw=0.25, alpha=0.5, zorder=2)
    axs[1].hist(lib_scores, bins=bins, label='Enamine and Zinc\nLibraries',
                color='tab:gray', edgecolor='k', lw=0.25, alpha=0.25)

    axs[1].set_yscale('log')
    b = axs[1].get_position()
    axs[1].set_position([b.x0, b.y0, b.width*0.5, b.height])
    axs[1].legend(title='Molecule Source', title_fontsize=8,
                  fontsize=7, bbox_to_anchor=[1,0.9])

    for i in range(2):
        axs[i].set_xlabel(r'Docking score / kcal mol$^{-1}$')
        axs[i].set_ylabel('Number of Molecules')
        axs[i].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    fig.savefig(outfile)


def plot_best_binders_new(df):
    '''
    Make figures of best binders using rdkit
    '''
    # Generated molecules
    gen = df[(df['source'].isin(['gen_mr1_filtered', 'gen_mr1_unfiltered_policy', 'gen_mr2_filtered'])) &
             (df['TotalEnergy'] < -170)]
    mols = [Chem.MolFromSmiles(i) for i in gen['smiles']]
    im = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=[2250, 1400], returnPNG=False)
    im.save('best_binders_generated.png')

    # Known binders
    known = df[(df['source'] == 'literature_binders') &
               (df['TotalEnergy'] < -130)]
    mols = [Chem.MolFromSmiles(i) for i in known['smiles']]
    im = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=[2250, 1400], returnPNG=False)
    im.save('best_binders_literature.png')

    # Probable binders
    probable = df[(df['source'].isin(['chembl', 'bindingdb'])) &
                  (df['TotalEnergy'] < -165)]
    mols = [Chem.MolFromSmiles(i) for i in probable['smiles']]
    im = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=[2250, 1400], returnPNG=False)
    im.save('best_binders_chembl_bindingdb.png')

    # Library
    library = df[(df['source'].isin(['enamine_hll', 'enamine_div', 'zinc'])) &
                  (df['TotalEnergy'] < -168)]
    mols = [Chem.MolFromSmiles(i) for i in library['smiles']]
    im = Draw.MolsToGridImage(mols, molsPerRow=6, subImgSize=[2250, 1400], returnPNG=False)
    im.save('best_binders_enamine_zinc.png')


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


def plot_phys_vs_docking_score(df, props_df, props, titles, outfile):
    '''
    Plot relationships between physical properties of the molecules and
    their docking scores
    '''
    fig, axes = plt.subplots(2, 4, dpi=300, figsize=[7.5, 3])
    axs = axes.ravel()
    for i in range(len(props)):
        ax = axs[i]
        prop = props[i]
        xs_all = props_df[prop].values
        ys_all = df['TotalEnergy'].values

        # Filter out nan's and do linear regression
        nans = np.isnan(xs_all) | np.isnan(ys_all)
        xs = xs_all[~nans]
        ys = ys_all[~nans]
        lr = stats.linregress(xs, ys)

        # Plot data
        ax.scatter(xs, ys, s=2, lw=0, alpha=0.8, color='tab:gray')

        # Plot line of best fit
        xs = [np.min(xs), np.max(xs)]
        ax.plot(xs, [lr.slope*i + lr.intercept for i in xs], '--k')

        ax.set_xlabel(titles[prop], fontsize=8)
        ax.set_ylabel('Docking Score /\n' + r'kcal mol$^{-1}$', fontsize=8)
        ax.set_title(f'r = {lr.rvalue:.2f}, p = {lr.pvalue:.2f}', fontsize=8)
        ax.tick_params(axis='both', labelsize=6)
        ax.spines[['top', 'right']].set_visible(False)
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


def calc_all_pcas(props_df, mfps):
    pca_results = {}

    # Calculate PCA on physical properties
    data_props_all = np.array(props_df.values[:,2:]).astype(np.float32)
    data_props = center_standardize(data_props_all)
    props_pca_proj, props_pca_exp_var = calc_pca(8, data_props)
    pca_results.update({'phys_prop' : {'proj' : props_pca_proj,
                                       'exp_var' : props_pca_exp_var}})
    # Calculate PCA on Morgan fingerprints
    for params in mfps.keys():
        data_all = mfps[params]
        data = center_standardize(data_all)
        fps_pca_proj, fps_pca_exp_var = calc_pca(100, data)
        pca_results.update({f'mfp_{params}' : {'proj' : fps_pca_proj,
                                               'exp_var' : fps_pca_exp_var}})

    return pca_results



def plot_pca_compare(pca_results, df, outfile):
    fig, axes = plt.subplots(2, 2, figsize=[7.5, 6], dpi=300)
    axs = axes.ravel()
    pcas = list(pca_results.keys())

    titles = {'phys_prop' : 'Vectors of Physical Properties',
              'mfp_r2_1024bits' : 'Morgan Fingerprints (r=2, bits=1024)',
              'mfp_r4_1024bits' : 'Morgan Fingerprints (r=4, bits=1024)',
              'mfp_r4_2048bits' : 'Morgan Fingerprints (r=4, bits=2048)'}
    sources = {'zinc' : 'ZINC',
               'enamine_div' : 'Enamine Diversity',
               'enamine_hll' : 'Enamine Hit Locator Library',
               'chembl' : 'ChEMBL',
               'bindingdb' : 'BindingDB',
               'literature_binders' : 'Literature',
               'gen_mr1_filtered' : 'Gen (Max Rxn 1)',
               'gen_mr1_unfiltered_policy' : 'Gen (Max Rxn 1,\nwith Policy)',
               'gen_mr2_filtered' : 'Gen (Max Rxn 2)'}
    colors = {'zinc' : 'lightgray',
              'enamine_div' : 'tab:green',
              'enamine_hll' : 'lightblue',
              'chembl' : 'tab:red',
              'bindingdb' : 'tab:orange',
              'literature_binders' : 'tab:pink',
              'gen_mr1_filtered' : 'k',
              'gen_mr1_unfiltered_policy' : 'tab:purple',
              'gen_mr2_filtered' : 'tab:brown'}

    for i in range(len(axs)):
        ax = axs[i]
        pca_type = pcas[i]
        pc1_var = pca_results[pca_type]['exp_var'][0]
        pc2_var = pca_results[pca_type]['exp_var'][1]

        for source in list(sources.keys()):
            indices = df.index[df['source'] == source].to_numpy()
            pc1_proj = pca_results[pca_type]['proj'][indices,0]
            pc2_proj = pca_results[pca_type]['proj'][indices,1]

            # Larger marker for generated molecules
            if source.startswith('gen'):
                s = 6
            else:
                s = 4
            ax.scatter(pc1_proj, pc2_proj, label=sources[source],
                       color=colors[source], alpha=0.8, s=s, lw=0)

        ax.set_title(titles[pca_type], fontsize=10)
        ax.set_xlabel(f'PC 1 ({100*pc1_var:.2f} % of Variance)')
        ax.set_ylabel(f'PC 2 ({100*pc2_var:.2f} % of Variance)')

        if pca_type == 'phys_prop':
            ax.set_xlim(-7, 30) # one outlier from chembl around (90, 30)
            ax.set_ylim(-15, 20)
            # Legend in first plot only
            ax.legend(loc='upper left', fontsize=6, ncols=2)
    plt.tight_layout()
    fig.savefig(outfile)


def calc_umap(data):
    '''
    Fit UMAP on 25% of data (for time reasons) then transform all of the data
    to get the final embedding
    '''
    umap = UMAP(n_components=2, n_neighbors=50, min_dist=1)
    num_molecs = np.shape(data)[0]
    indices = np.random.choice(np.arange(num_molecs),
                               size=np.ceil(0.25*num_molecs).astype(int))
    data_subset = data[indices,:]
    print("fitting...", data_subset.shape)
    umap_embed = umap.fit(data_subset)
    print("embedding...")
    umap_embedding_full = umap.transform(data)
    return umap_embedding_full


def calc_all_umaps(pca_results):
    '''
    Calculate UMAPs on PCAs (all 4 types)
    Doing PCA first to remove noise and reduce dimensionality
    '''
    pcas = list(pca_results.keys())
    umaps = {}

    # Calculate UMAP on all PCAs
    for pca_type in pcas:
        pcs = pca_results[pca_type]['proj']
        umap_embedding_full = calc_umap(pcs)
        umaps.update({pca_type : umap_embedding_full})
    return umaps


def plot_umaps_compare(umap_results, df, outfile):
    fig, axes = plt.subplots(2, 2, figsize=[7.5, 6], dpi=300)
    axs = axes.ravel()
    pcas = list(pca_results.keys())

    titles = {'phys_prop' : 'Vectors of Physical Properties',
              'mfp_r2_1024bits' : 'Morgan Fingerprints (r=2, bits=1024)',
              'mfp_r4_1024bits' : 'Morgan Fingerprints (r=4, bits=1024)',
              'mfp_r4_2048bits' : 'Morgan Fingerprints (r=4, bits=2048)'}
    sources = {'zinc' : 'ZINC',
               'enamine_div' : 'Enamine Diversity',
               'enamine_hll' : 'Enamine Hit Locator Library',
               'chembl' : 'ChEMBL',
               'bindingdb' : 'BindingDB',
               'literature_binders' : 'Literature',
               'gen_mr1_filtered' : 'Gen (Max Rxn 1)',
               'gen_mr1_unfiltered_policy' : 'Gen (Max Rxn 1,\nwith Policy)',
               'gen_mr2_filtered' : 'Gen (Max Rxn 2)'}
    colors = {'zinc' : 'lightgray',
              'enamine_div' : 'tab:green',
              'enamine_hll' : 'lightblue',
              'chembl' : 'tab:red',
              'bindingdb' : 'tab:orange',
              'literature_binders' : 'tab:pink',
              'gen_mr1_filtered' : 'k',
              'gen_mr1_unfiltered_policy' : 'tab:purple',
              'gen_mr2_filtered' : 'tab:brown'}

    for i in range(len(axs)):
        ax = axs[i]
        umap_type = pcas[i]
        
        for source in list(sources.keys()):
            indices = df.index[df['source'] == source].to_numpy()
            umap1 = umap_results[umap_type][indices,0]
            umap2 = umap_results[umap_type][indices,1]
        
            # Larger marker for generated molecules
            if source.startswith('gen'):
                s = 6
            else:
                s = 4
            ax.scatter(umap1, umap2, label=sources[source],
                       color=colors[source], alpha=0.8, s=s, lw=0)

        ax.set_title(titles[umap_type], fontsize=10)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

    plt.tight_layout()
    fig.savefig(outfile)


groups = ['library_molecs', 'gen_mr1_filtered', 'gen_mr1_unfiltered_policy',
          'gen_mr2_filtered', 'literature_binders']
smiles_path = '../../data/smiles'
gen_path = '../../SyntheMol_generated_candidate_molecules'
dock_path = '../../docking_results'
reps_path = '../../data/representations'

# Load all data and combine
df = get_smiles_sources_scores(groups, smiles_path, gen_path, dock_path)
props_df, mfps = get_representations(groups, reps_path)

# Check shapes
assert len(df) ==  len(props_df) == np.shape(mfps['r2_1024bits'])[0]

# Plot distributions of docking scores
plot_docking_score_distr_comparisons(df, outfile='docking_score_distrs_w_comparison.png')

# Make figures of the best binders from each group
plot_best_binders_new(df)

# Plot distributions of physical properties
props = [i for i in props_df.columns if i not in ['smiles', 'Unnamed: 0']]
titles = {'qed' : 'Quantitative Estimate\nof Druglikeness',
              'SPS' : 'Spatial Complexity\nScore',
              'MolWt' : 'Molecular Weight',
              'MaxPartialCharge' : 'Maximum Partial Charge',
              'MinPartialCharge' : 'Minimum Partial Charge',
              'TPSA' : 'Total Polar Surface Area',
              'MolLogP' : 'LogP',
              'MolMR' : 'Molar Refractivity'}

plot_distrs_of_properties(props_df, props, titles, outfile='distr_phys_prop_1d_hist.png')
plot_distrs_of_properties_2d(df, props_df, props, titles, outfile='distr_phys_prop_2d_hist.png')

# Are physical properties correlated with docking score?
plot_phys_vs_docking_score(df, props_df, props, titles, outfile='linear_regression_phys_dock_score.png')

# Calculate PCAs
pca_results = calc_all_pcas(props_df, mfps)

# Plot PCAs
plot_pca_compare(pca_results, df, outfile='pca_comparison_all.png')

# Calculate UMAPs
umaps = calc_all_umaps(pca_results)

# Plot UMAPs
plot_umaps_compare(umaps, df, outfile='umap_compare_all.png')
