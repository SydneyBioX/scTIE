# scTIE: data integration and inference of gene regulation using single-cell temporal multimodal data

scTIE (single-cell Temporal Integration and inference of multimodal Experiments) is an autoencoder-based method for integrating multimodal profiling of scRNA-seq and scATAC-seq data over a time course and inferring cell-type specific GRNs. scTIE projects cells from all time points into a common embedding space, followed by extracting interpretable information from this space to predict cell trajectories. 

scTIE is developed and tested using PyTorch 1.9.1.

## Tutorials

A step-by-step tutorial using a HSPCs 10x multiome data provided in Kaggle and NIPS2022 competition "Open Problems - Multimodal Single-Cell Integration is demonstrated here: [link](https://github.com/SydneyBioX/scTIE/blob/main/tutorial/Analysis%20of%20HSPCs%20data%20using%20scTIE.ipynb), the data can be donwloaded from [link](https://www.maths.usyd.edu.au/u/yingxinl/wwwnb/scTIE/NIPS.zip).

## Installation

scTIE can be obtained by simply clonning the github repository:

```
git clone https://github.com/SydneyBioX/scTIE.git
```

The following python packages are required to be installed before running scTIE:
`torch`, `numpy`, `os`, `random`, `time`, `datetime`, `scipy`, `pandas`, `argparse`, `csv` and `shutil`.


## Reparing input for scTIE

**Input files**

scTIE's main function requires one RNA gene expression and ATAC peak matrix in `.npz` format for each day. To prepare the input for scTIE, modifying dataset paths in `process_db.py` which takes `.h5` files of expression matrix stored in `matrix/data` as input and generate `.npz` files for each expression matrix.

**Calculating initial growth rate**

The optimal transport calculation in scTIE requires a list of initial growth rate files as input. This can be generated via `calculate_growth_rate.py` with modification of the data directory paths. 

**Modifying `config.py` file**

The following setting may require modification:

+ `rna_size`: Number of genes in RNA data.
+ `atac_size`: Number of peaks in ATAC data.
+ `days`: An array indicates the day of the data.
+ `growth_estimates`: An array indicates a list of paths of initial growth rate for each day.
+ `rna_paths`: An array indicates a list of paths of RNA gene expression in `.npz` format.
+ `atac_paths`: An array indicates a list of paths of ATAC peak expression in `.npz` format.
+ `rna_input`: A string indicates whether the RNA input is `counts` or `logcounts` data.

## Running scTIE


[Part 1: Temporal multimodal data integration](#part1)  
[Part 2: Embedding gradient backpropagation](#part2)  
[Part 3: Cell transition probability backpropagation](#part3)  

<a name="part1"/>

### Part 1: Temporal multimodal data integration

Run `main.py` to perform the autoencoder-based data integration of the temporal multimodal data.


**Arguments:**

- `DB`: The name of the dataset.
- `modalities`: A list of cell measurement modalities to train on.
- `ckpt`: The previous RNA pretrained model path.
- `days`: An array indicates the day of the data.
- `hidden_size`: The size of the hidden layer, set as 1000 by default.
- `embedding_size`: The size of the embedding, set as 64 by default.
- `batch_size`: Batch size, set as 256 by default.
- `lr`: Learning rate, set as 0.1 by default.
- `epochs`: The total training epochs, set as 500 by default.
- `ot_epochs`: The frequencies of update of the optimal transport matrices, set as 10 by default.
- `anchor_epochs`: The total epochs where RNA model is fixed and the other (eg, ATAC) is updated. The number of joint training epochs is therefore `epochs - anchor_epochs`, set as 300 by default.
- `common_loss_weight`: The weight of modality alignment, set as 1 by default.
- `restored_loss_weight`: The weight of the reconstruction, set as 10 by default.
- `ot_weight`: The weight of OT loss, set as 0.1 by default.
- `ot_growth_iters`: The number of iteraction in OT, set as 3 by default.
- `output_dir`: Folder to save the computed embedding gradient.
- `seed`: Random seed for reproducibility, set `None` for random training.
- `ncores`: Number of cores used.

**Detailed training procedures:**

Stage 1: Train RNA only 

1. Set `modalities` to `['rna']` (RNA only).
2. Run `main.py`.
3. The pretrained RNA modal is stored in `<output_dir>/<DB>/`.
    
Stage 2: Train RNA and ATAC jointly:

1. Set `ckpt` to previous RNA pretrained model path.
2. Set `modalities` to `['rna', 'atac']` (RNA and ATAC).
3. Run `main.py`.


**Output:**

After running the script, in the `<output_dir>/<exp_name>/`, there will be the following output files:

1. Embedding for each day:

+ RNA embedding: `rna_embeddings_<i>_<epoch>.npy`
+ ATAC embedding: `atac_embeddings_<i>_<epoch>.npy`
+ Common embedding: `common_embeddings_<i>_<epoch>.npy`

where `i` refers to the day ID (0, 1, 2, 3, ...) and `epoch` refers to the saved epoch.


2. Reconstruction data for each day:

+ RNA reconstruction: `rna_restored_<i>_<epoch>.npy`
+ ATAC reconstruction: `atac_restored_<i>_<epoch>.npy`

where `i` refers to the day ID (0, 1, 2, 3, ...) and `epoch` refers to the saved epoch.
    
3. Optimal transport matrices between each pair of day:

+ `common_ganna_day<d1>_day<d2>_wad.npy`



**Examples:**

```
# Stage 1
python3 main.py --modalities rna  \
  --DB "nips" \
  --days 2 3 4 7 \
  --ot_epochs 10 \
  --hidden_size 1000 \
  --embedding_size 64 \
  --batch_size 256 \
  --lr 0.1 \
  --epochs 500 \
  --anchor_epochs 300 \
  --common_loss_weight 1 \
  --restored_loss_weight 10 \
  --ot_weight 0.1 \
  --ot_growth_iters 3 \
  --output_dir "output/NIPS" \
  --ncores 1

# Stage 2
python3 main.py --modalities rna atac \
  --DB "nips" \
  --days 2 3 4 7 \
  --ckpt "output/NIPS/checkpoint_rna_nips_mse_d2-3-4-7_ot0.1_common1.0_otepoch10_hidden1000_e500.pth.tar" \
  --ot_epochs 10 \
  --hidden_size 1000 \
  --embedding_size 64 \
  --batch_size 256 \
  --lr 0.1 \
  --epochs 500 \
  --anchor_epochs 300 \
  --common_loss_weight 1 \
  --restored_loss_weight 10 \
  --ot_weight 0.1 \
  --ot_growth_iters 3 \
  --output_dir "output/NIPS" \
  --ncores 1
```


<a name="part2"/>

### Part 2: Embedding gradient backpropagation 

Run `embedding_grad_all.py` to backpropagate the gradient of each dimension in the embedding layer with respect to gene and peak input.


**Arguments:**

- `DB`: The name of the dataset.
- `days`: An integer indicates which days to calculate.
- `cell_set_idx_path`:The path of a set of cell id in the time point to compute the embedding gradient. If it is None, then all the cells on the day <day> will be calculated. The cell id starts from 0.
- `cell_set_name`:The name of the cell set.
- `average`: Whether to only output the average of the embedding gradient of a set of cell (provided by `cell_set_idx_path`). If set to `False`, then it will output the embedding gradient for each cell.
- `pretrained_name`: Pretrained model checkpoint name.
- `pretrained_checkpoints_dir`: Folder where pretrained models are saved.
- `save_dir`: Folder to save the computed embedding gradient.
- `seed`: Random seed for reproducibility, set `None` for random training.

**Output:**

After running the script, if the average is set to `False`, for each cell, two `.npy` files `jacob_rna_cell{cell_id}.npy` and `jacob_atac_cell{cell_id}.npy` will be saved to `<save_dir>/day<day>/` folder. 

If the average is set to `True`, only two `.npy` files `jacob_rna_cellset_{cell_set_name}.npy` and `jacob_atac_cellset_{cell_set_name}.npy` will be saved to `<save_dir>/day<day>/` folder, which calculate the average of the embedding gradients of all the cells. If there are multiple days provided in the `days` argument, then the average results will be saved in `<save_dir>`.
   
The shape of each jacobian J is (emb_size, input_size).


**Examples:**

1. Output the embedding gradient for all cells for a specific day.

The following codes ran in terminal will output two `.npy` files `jacob_rna_cell{cell_id}.npy` and `jacob_atac_cell{cell_id}.npy` for all the cells on Day 2, saved in  `output/NIPS/emb_grad_all/day2/` folder. 

```
python3 embedding_grad_all.py \
--DB "nips" \
--pretrained_name "rna-atac_nips_mse_d2-3-4-7_ot0.1_common1.0_otepoch10_hidden1000_rna-pretrain_e500" \
--pretrained_checkpoints_dir "output/NIPS/" \
--save_dir "output/NIPS/emb_grad_all/" \
--days 2
```

2. Output the average embedding gradient for a selected cell set

The following codes ran in terminal will output two `.npy` files `jacob_rna_cellset_MoP_average.npy` and `jacob_atac_cellset_MoP_average.npy` for a set of cells on Day 3, whose indexes are indicated by `data/NIPS/day3_MoP_id_list.txt`. The output is saved in  `output/NIPS/emb_grad_all/day3/` folder. 

```
python3 embedding_grad_all.py \
--DB "nips" \
--cell_set_idx_path "data/NIPS/day3_MoP_id_list.txt" \
--cell_set_name "MoP" \
--average \
--pretrained_name "rna-atac_nips_mse_d2-3-4-7_ot0.1_common1.0_otepoch10_hidden1000_rna-pretrain_e500" \
--pretrained_checkpoints_dir "output/NIPS/" \
--save_dir "output/NIPS/emb_grad_all/" \
--days 3
```

3. Output the average embedding gradient for cells from multiple time points

The following codes ran in terminal will output two `.npy` files `jacob_rna_cellset_all_average.npy` and `jacob_atac_cellset_all_average.npy` for all the cells on Day 2, 3, 4 & 7. The output is saved in  `output/NIPS/emb_grad_all/` folder. 

```
python3 embedding_grad_all.py \
--DB "nips" \
--cell_set_name "all" \
--average \
--pretrained_name "rna-atac_nips_mse_d2-3-4-7_ot0.1_common1.0_otepoch10_hidden1000_rna-pretrain_e500" \
--pretrained_checkpoints_dir "output/NIPS/" \
--save_dir "output/NIPS/emb_grad_all/" \
--days 2 3 4 7
```

<a name="part3"/>

### Part 3: Cell transition probability backpropagation


Run `infer_deconv.py` to backpropagate feature gradients from cell transition probability prediction. 


**Preparing the input**

Apart from some output in the previous steps, this function requires the following extra input files:


1. A `.csv` file indicates the indices of peaks to be removed (Optional, but suggested).
2. A few `.csv` files indicate the **Selection of three groups of cells ($G_0, G_1, G_2$)**

+ A group of cells from earlier days ($G_0$): This is indicated by `day{d1}_{day1_cluster}.csv`, `d in {day1}`, where the each `.csv` file should indicate the indices of cluster `day1_cluster` on day `d1`.
+ Two groups of cells from later days ($G_1, G_2$): This is indicated by `day{d2}_{day2_cluster}_from_day{d1}_{day1_cluster}.csv`, `d in {day1}`, where the each `.csv` file should indicate the indices of descendants of day `d1` `day1_cluster` that are cluster `day2_cluster` on day `d2`. `day2_cluster` should contain two groups for contrast.

*Examples:* 

If we would like to look at what are the features that are associated with the transition probability of HSC on Day 2, 3 to NeuP on Day 3, 4, 7, compared to all other descendants, we need to set the parameters as the following:
```
--day1 2 3 \
--day2 3 4 7 \
--day1_cluster "HSC" \
--day2_clusters "NeuP" "NeuPOthers" \
```
and it requires the following `.csv` file in `<input_dir>`:

+ $G_0$: `day2_HSC.csv`, `day3_HSC.csv`
+ $G_1$: `day3_NeuP_from_day2_HSC.csv`, `day4_NeuP_from_day2_HSC.csv`, `day7_NeuP_from_day2_HSC.csv`,  `day4_NeuP_from_day3_HSC.csv`, `day7_NeuP_from_day3_HSC.csv`
+ $G_2$: `day3_NeuPOthers_from_day2_HSC.csv`, `day4_NeuPOthers_from_day2_HSC.csv`, `day7_NeuPOthers_from_day2_HSC.csv`,  `day4_NeuPOthers_from_day3_HSC.csv`, `day7_NeuPOthers_from_day3_HSC.csv`


**Arguments:**

- `DB`: The name of the dataset.
- `day1`: An array indicates the day of `day1_cluster`, a group of cells on earlier days.
- `day2`: An array indicates the day of `day2_cluster`, two group of cells on later days, which are the potential descendants of `day1_cluster`.
- `day1_cluster`: A string indicates the starting cluster on earlier days `day1`.
- `day2_cluster`: An array indicates the two ending clusters on later days `day2`.
- `input_dir`: Folder where input cluster indices csv files are stored.
- `epochs`: The total training epochs, set as 200 by default.
- `lr`: Learning rate, set as 0.001 by default.
- `clip_max_probs`: Minimum probability (confidence) threshold. Remove all training data under this threshold, set as 0 by default.
- `common_loss_weight`: The weight of modality alignment, set as 0.1 by default.
- `atac_l1_weight`: The weight for ATAC encoder L1 regularization loss.
- `rna_l1_weight`: The weight for RNA encoder L1 regularization loss.
- `pretrained_name`: Pretrained model name.
- `pretrained_checkpoints_dir`: Folder where pretrained model weights are stored.
- `pretrained_results_dir`: Folder where pretrained model transport matrices are stored.
- `filter_peaks`: Path of a list of peaks indices being filtered out. 
- `output_dir`: Folder to save the computed embedding gradient.
- `seed`: Random seed for reproducibility, set `None` for random training.




**Output:**

After running the script, in the `<output_dir>/<exp_name>`, there will be the following output files:


1. Transition probabilities to `day2_cluster` from `day1_cluster` on each `day1`: `prob_day<d>_<day1_cluster>.npy`.
2. Average of gradients for RNA and ATAC: `grad_atac.npy` (A vector of the size of ATAC) and `grad_rna.npy` (A vector of the size of RNA). A positive gradient for gene (or peak) means increasing the input feature value tend to increase the cells' probabilities of becoming $G_1$, while a negative value indicates more contribution to $G_2$. 

**Examples:**

The following codes look at what are the features that are associated with the transition probability of HSC on Day 2, 3 to NeuP on Day 3, 4, 7, compared to all other descendants:
```
python3 infer_deconv.py \
--DB "nips" \
--day1 2 3 \
--day2 3 4 7 \
--day1_cluster "HSC" \
--day2_clusters "NeuP" "NeuPOthers" \
--input_dir data/NIPS/cluster_idx/ \
--atac_l1_weight 100 \
--rna_l1_weight 100 \
--output_dir output/NIPS/grad_results/ \
--filter_peaks data/NIPS/cluster_idx/remove_peaks_by_accessibility_idx_prop_1.1.csv \
--pretrained_results_dir output/NIPS/ \
--pretrained_checkpoints_dir output/NIPS/ \
--pretrained_name "rna-atac_nips_mse_d2-3-4-7_ot0.1_common1.0_otepoch10_hidden1000_rna-pretrain" \
--seed 1
```

