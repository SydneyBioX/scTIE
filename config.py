from argparse import ArgumentParser
import torch

class Config(object):
    def __init__(self, DB = "multiome_5pct", days = None, ot_epochs = 10, 
    modalities = ["rna"], ckpt = None, rna_input = "counts",
    hidden_size = 1000, embedding_size = 64, batch_size = 256, lr = 0.1, 
    epochs = 500, anchor_epochs = 300, common_loss_weight = 1, 
    restored_loss_weight = 10, ot_weight = 0.1, ot_growth_iters = 3, 
    output_dir = "output/", seed = 6666, ncores = 0):
        self.use_cuda = True
        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
        self.loss = "mse"  # bce or mse
        self.ot_epochs = ot_epochs
        self.modalities = modalities
        self.ckpt = ckpt
        self.seed = seed #int or None
        self.ncores = ncores
        self.ot_growth_iters = ot_growth_iters
        self.days = days  # days
        self.output_dir = output_dir
        if DB == "multiome_5pct":
            # DB info
            self.rna_size = 26717
            self.atac_size = 61744
            if days is None:
              self.days = [2, 4, 6]
            self.growth_estimates = [f"data/data_multiome_processed/day{day}_growth_gs_init.npy" for day in self.days]
            self.rna_paths = [
                f"data/data_multiome_processed/rna_counts/sce_mESC_multiome_day{d}_rna_counts.npz" for d in self.days
            ]
            self.atac_paths = [
                f"data/data_multiome_processed/atac_5pct/sce_mESC_multiome_day{d}_atac_5pctExprs.npz"
                for d in self.days
            ]
            self.rna_input = "counts"

          
        if DB == "nips":
            # DB info
            self.rna_size = 18152
            self.atac_size = 61794
            if days is None:
              self.days = [2, 3, 4, 7]
            self.growth_estimates = [f"data/NIPS/day{day}_growth_gs_init.npy" for day in self.days]
            self.rna_paths = [f"data/NIPS/nips_day{d}_rna.npz" for d in self.days]
            self.atac_paths = [f"data/NIPS/nips_day{d}_atac.npz" for d in self.days]
            self.rna_input = "logcounts"
            
        if DB == "nips_full":
            # DB info
            self.rna_size = 18152
            self.atac_size = 61816
            if days is None:
              self.days = [2, 3, 4, 7]
            self.growth_estimates = [f"data/NIPS_full/day{day}_growth_gs_init.npy" for day in self.days]
            self.rna_paths = [f"data/NIPS_full/sce_mESC_multiome_day{d}_rna.npz" for d in self.days]
            self.atac_paths = [f"data/NIPS_full/sce_mESC_multiome_day{d}_atac.npz" for d in self.days]
            self.rna_input = "logcounts"
        

        self.rna_labels = [None] * len(self.days)
        self.atac_labels = [None] * len(self.days)
        self.hk_gene_path = None  # path to housekeeping gene indices
        # Training config
        self.hidden_size = hidden_size  # Dimension for hidden AE layer
        self.embedding_size = embedding_size  # Dimension of embedding

        self.batch_size = batch_size # training batch size
        self.lr = lr # selection 0.1 (rna), 0.1 (atac) 
        self.lr_decay_epoch = 60
        self.epochs = epochs
        self.save_epochs = epochs
        self.anchor_epochs = anchor_epochs
        # Number of epochs in joint training to fix RNA weights (anchoring) and train ATAC only.
        self.momentum = 0.9

        self.common_loss_weight = common_loss_weight # Alignement weight (RNA - ATAC)
        self.restored_loss_weight = restored_loss_weight # Restore loss weight
        self.ot_weight = ot_weight # Optimal transport loss weight
       
        # Experiment ID name
        days = '-'.join([str(d) for d in self.days])
        modalities = '-'.join([str(m) for m in self.modalities])
        self.exp_name =  f'{modalities}_{DB}_{self.loss}_d{days}_ot{self.ot_weight}_common{self.common_loss_weight}_otepoch{self.ot_epochs}_hidden{self.hidden_size}'
        if self.ckpt is not None: # Load previous RNA pretrain model
            self.exp_name = f'{self.exp_name}_rna-pretrain'



