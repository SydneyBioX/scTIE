import numpy as np
import scipy.sparse
import scipy.stats
import os
import pandas as pd
import h5py




# Below codes are wot
# apply logistic function to transform to birth rate and death rate
def logistic(x, L, k, x0=0):
    f = L / (1 + np.exp(-k * (x - x0)))
    return f
def gen_logistic(p, beta_max, beta_min, pmax, pmin, center, width):
    return beta_min + logistic(p, L=beta_max - beta_min, k=4 / width, x0=center)

def beta(p, beta_max=1.7, beta_min=0.3, pmax=1.0, pmin=-0.5, center=0.25):
    return gen_logistic(p, beta_max, beta_min, pmax, pmin, center, width=0.5)

def delta(a, delta_max=1.7, delta_min=0.3, amax=0.5, amin=-0.4, center=0.1):
    return gen_logistic(a, delta_max, delta_min, amax, amin, center,
                          width=0.2)
                          
def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


# from http://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
def fdr(pvals, is_sorted=False, method='indep'):
    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1. / np.arange(1, len(pvals_sorted) + 1))  # corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
    ##    elif method in ['n', 'negcorr']:
    ##        cm = np.sum(np.arange(len(pvals)))
    ##        ecdffactor = ecdf(pvals_sorted)/cm
    else:
        raise ValueError('only indep and negcorr implemented')

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected > 1] = 1
    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        return pvals_corrected_
    else:
        return pvals_corrected


def get_p_value_ci(n, n_s, z):
    # smooth
    n = n + 2
    n_s = n_s + 1
    n_f = n - n_s
    ci = (z / n) * np.sqrt((n_s * n_f) / n)
    return ci


def read_gmx(path, feature_ids=None):
    with open(path) as fp:
        set_ids = fp.readline().split('\t')
        descriptions = fp.readline().split('\t')
        nsets = len(set_ids)
        for i in range(len(set_ids)):
            set_ids[i] = set_ids[i].rstrip()

        row_id_lc_to_index = {}
        row_id_lc_to_row_id = {}
        x = None
        array_of_arrays = None
        if feature_ids is not None:
            for i in range(len(feature_ids)):
                fid = feature_ids[i].lower()
                row_id_lc_to_index[fid] = i
                row_id_lc_to_row_id[fid] = feature_ids[i]
            x = np.zeros(shape=(len(feature_ids), nsets), dtype=np.int8)
        else:
            array_of_arrays = []
        for line in fp:
            tokens = line.split('\t')
            for j in range(nsets):
                value = tokens[j].strip()
                if value != '':
                    value_lc = value.lower()
                    row_index = row_id_lc_to_index.get(value_lc)
                    if feature_ids is None:
                        if row_index is None:
                            row_id_lc_to_row_id[value_lc] = value
                            row_index = len(row_id_lc_to_index)
                            row_id_lc_to_index[value_lc] = row_index
                            array_of_arrays.append(np.zeros(shape=(nsets,), dtype=np.int8))
                        array_of_arrays[row_index][j] = 1
                    elif row_index is not None:
                        x[row_index, j] = 1
        if feature_ids is None:
            feature_ids = np.empty(len(row_id_lc_to_index), dtype='object')
            for rid_lc in row_id_lc_to_index:
                feature_ids[row_id_lc_to_index[rid_lc]] = row_id_lc_to_row_id[rid_lc]

        # if array_of_arrays is not None:
        #     x = pd.DataFrame(array_of_arrays, index = set_ids)
        # obs = pd.DataFrame(index=feature_ids)
        # var = pd.DataFrame(data={'description': descriptions},
        #     index=set_ids)
        x = pd.DataFrame(x, index = feature_ids, columns = set_ids)
        return x
        #return anndata.AnnData(x, obs=obs, var=var)


def convert_binary_dataset_to_dict(ds):
    cell_sets = {}
    for i in range(ds.shape[1]):
        selected = np.where(ds[:, i].X == 1)[0]
        cell_sets[ds.var.index[i]] = list(ds.obs.index[selected])
    return cell_sets


def get_filename_and_extension(name):
    name = os.path.basename(name)
    dot_index = name.rfind('.')
    ext = ''
    basename = name
    if dot_index != -1:
        ext = name[dot_index + 1:].lower()
        basename = name[0:dot_index]
        if ext == 'txt':  # check for .gmt.txt e.g.
            dot_index2 = basename.rfind('.')
            if dot_index2 != -1:
                ext2 = basename[dot_index2 + 1:].lower()
                if ext2 in set(['gmt', 'grp', 'gmx']):
                    basename = basename[0:dot_index2]
                    return basename, ext2
    return basename, ext

def read_sets(path, feature_ids=None, as_dict=False):
    path = str(path)
    hash_index = path.rfind('#')
    set_names = None
    if hash_index != -1:
        set_names = path[hash_index + 1:].split(',')
        path = path[0:hash_index]
    ext = get_filename_and_extension(path)[1]
    if ext == 'gmt':
        gs = read_gmt(path, feature_ids)
    elif ext == 'gmx':
        gs = read_gmx(path, feature_ids)
    elif ext == 'txt' or ext == 'grp':
        gs = read_grp(path, feature_ids)
    else:
        raise ValueError('Unknown file format "{}"'.format(ext))
    if set_names is not None:
        gs_filter = gs.var.index.isin(set_names)
        gs = gs[:, gs_filter]
    if as_dict:
        return convert_binary_dataset_to_dict(gs)
    return gs


def score_gene_sets(x, gs, method='mean_z_score', max_z_score=5, permutations=None,
                    random_state=0, smooth_p_values=True):
    """Score gene sets.
    Note that datasets and gene sets must be aligned prior to invoking this method. No check is done.
    mean_z_score: Compute the z-score for each gene in the set. Truncate these z-scores at 5 or −5, and define the signature of the cell to be the mean z-score over all genes in the gene set.
    Parameters
    ----------
    random_state : `int`, optional (default: 0)
        The random seed for sampling.
    Returns
    -------
    Observed scores and permuted p-values if permutations > 0
    """

    if permutations is None:
        permutations = 0
        
    # x = ds.X
    gs_1_0 = gs
    if not scipy.sparse.issparse(gs) and len(gs.shape) == 1:
        gs_1_0 = np.array([gs_1_0]).T

    if not scipy.sparse.issparse(gs_1_0):
        gs_1_0 = scipy.sparse.csr_matrix(gs_1_0)

    gs_indices = (gs_1_0 > 0)
    if hasattr(gs_indices, 'toarray'):
        gs_indices = gs_indices.toarray()
    gs_indices = gs_indices.flatten()

    if len(x.shape) == 1:
        x = np.array([x])
    # preprocess the dataset
    if method == 'mean_z_score':
        x = x[:, gs_indices]  # only include genes in gene set
        if scipy.sparse.issparse(x):
            x = x.toarray()
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        std = np.sqrt(var)
        # std[std == 0] = 1e-12 # avoid divide by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (x - mean) / std
        x[np.isnan(x)] = 0
        x[x < -max_z_score] = -max_z_score
        x[x > max_z_score] = max_z_score
    elif method == 'mean_rank':  # need all genes for ranking
        ranks = np.zeros(x.shape)
        is_sparse = scipy.sparse.issparse(x)
        for i in range(x.shape[0]):  # rank each cell separately
            row = x[i, :]
            if is_sparse:
                row = row.toarray()
            ranks[i] = scipy.stats.rankdata(row, method='min')
        x = ranks
        x = x[:, gs_indices]
    else:
        x = x[:, gs_indices]
        if scipy.sparse.issparse(x):
            x = x.toarray()
    observed_scores = x.mean(axis=1)
    if hasattr(observed_scores, 'toarray'):
        observed_scores = observed_scores.toarray()

    # gene sets has genes on rows, sets on columns
    # ds has cells on rows, genes on columns
    # scores contains cells on rows, gene sets on columns

    if permutations is not None and permutations > 0:
        if random_state:
            np.random.seed(random_state)
        p_values = np.zeros(x.shape[0])
        permuted_X = x.T.copy()  # put genes on rows to shuffle each row indendently
        for i in range(permutations):
            for _x in permuted_X:
                np.random.shuffle(_x)
            # count number of times permuted score is >= than observed score
            p_values += permuted_X.mean(axis=0) >= observed_scores

        k = p_values
        if smooth_p_values:
            p_values = (p_values + 1) / (permutations + 2)
        else:
            p_values = p_values / permutations
        return {'score': observed_scores, 'p_value': p_values, 'fdr': fdr(p_values), 'k': k}

    return {'score': observed_scores}
  



def calculate_gr(file_name, GENE_SETS_PATH = "data/gene_sets.gmx"):

  h5 = h5py.File(file_name, "r")
  genes_name = np.array(h5py.Dataset.asstr(h5["matrix/features"]))

  data = h5["matrix/data"]
  gs = read_sets(GENE_SETS_PATH, genes_name)
  gs = gs[['Cell.cycle', 'Apoptosis']]
  gene_set_scores_df = pd.DataFrame(index=np.arange(data.shape[1]))
  data = np.transpose(data)
  for j in range(gs.shape[1]):
    gene_set_name = str(gs.columns[j])
    result = score_gene_sets(x = data, gs = gs[gene_set_name], permutations = 0, method = 'mean_z_score')
    gene_set_scores_df[gene_set_name] = result['score']
  
  proliferation = gene_set_scores_df['Cell.cycle']
  apoptosis = gene_set_scores_df['Apoptosis']
  
  
  birth = beta(proliferation)
  death = delta(apoptosis)
  
  # growth rate is given by 
  gr = np.exp(birth-death)
  gr = np.array(gr)
  #growth_rates_df = pd.DataFrame(index=gene_set_scores_df.index, data={'cell_growth_rate':gr})
  return gr



if __name__ == "__main__":
  days = [2, 3, 4, 7]
  data_dir = "data/NIPS/"
  for d in days:
    print(d)
    file_name = f"{data_dir}/nips_day{d}_rna.h5"
    growth_rates_df = calculate_gr(file_name)
    np.save(f"{data_dir}/day{d}_growth_gs_init.npy", growth_rates_df)


