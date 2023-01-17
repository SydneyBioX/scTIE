import torch
import torch.nn as nn
import torch.optim as optim
from shutil import copyfile
from torch.autograd import Variable
import numpy as np
from datetime import datetime

from util.dataloader import (
    PrepareUnprocessedDataloaderList,
    PrepareUnprocessedDataloader,
)
from util.model_residual import BN_common_encoder, BN_resid_encoder, BN_concat_decoder
from util.closs import L1regularization, PairLoss
from util.utils import *

from util.ot_solvers import get_total_ot_loss, solve_ot, compute_transport_map

class TrainingProcess:
    def __init__(self, config):
        self.config = config
        if not os.path.exists(f"{self.config.output_dir}/{self.config.exp_name}"):
            os.makedirs(f"{self.config.output_dir}/{self.config.exp_name}")

        # List of housekeeping genes. If specified, then the housekeeping genes dimensions
        # will be removed from the training data
        self.housekeeping_genes_idx = None
        if self.config.hk_gene_path is not None:
            ids = np.genfromtxt(self.config.hk_gene_path, delimiter=",")[1:, 1].astype(np.int) - 1
            ids.sort()
            self.housekeeping_genes_idx = ids

        # load data
        self.train_loader, _ = PrepareUnprocessedDataloaderList(
            config.rna_paths,
            config.atac_paths,
            config.rna_labels,
            config.atac_labels,
            config.batch_size,
            self.housekeeping_genes_idx,
            config.rna_input,
            config.ncores,
        ).getloader()
        self.test_loaders = []
        for i in range(len(config.rna_paths)):
            _, test_loader = PrepareUnprocessedDataloader(
                [config.rna_paths[i]],
                [config.atac_paths[i]],
                [config.rna_labels[i]],
                [config.atac_labels[i]],
                config.batch_size,
                self.housekeeping_genes_idx,
                config.rna_input,
                config.ncores,
            ).getloader()
            self.test_loaders.append(test_loader)

        # Remove housekeeping gene dimensions if specified
        if self.housekeeping_genes_idx is not None:
            config.rna_size -= len(self.housekeeping_genes_idx)

        # initialize dataset
        self.model_common_encoder = BN_common_encoder(
            config.rna_size, config.atac_size, config.embedding_size, config.hidden_size
        ).cuda()
        self.model_common_encoder = self.model_common_encoder.float()

        self.model_resid_encoder = BN_resid_encoder(
            config.rna_size, config.atac_size, config.embedding_size, config.hidden_size
        ).cuda()
        self.model_resid_encoder = self.model_resid_encoder.float()

        self.model_decoder = BN_concat_decoder(
            config.embedding_size, config.rna_size, config.atac_size, config.hidden_size
        ).cuda()
        self.model_decoder = self.model_decoder.float()

        # Use pretrained weights
        if self.config.ckpt is not None:
            print("Loading state dict")
            ckpt = torch.load(self.config.ckpt)
            self.model_common_encoder.load_state_dict(ckpt["model_encoding_state_dict"])
            self.model_decoder.load_state_dict(ckpt["model_decoding_state_dict"])

        # initialize criterion (loss)
        self.criterion_pair = PairLoss()
        self.recon_loss = nn.MSELoss()

        # initialize optimizer (sgd/momemtum/weight decay)
        self.optimizer = optim.SGD(
            [
                {"params": self.model_common_encoder.parameters()},
                {"params": self.model_resid_encoder.parameters()},
                {"params": self.model_decoder.parameters()},
            ],
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=0,
        )

        # OT-related
        self.rna_gammas = {}  # OT transport matrices
        self.atac_gammas = {}  # OT transport matrices

        # timeing
        self.total_ae_time = 0
        self.total_accum_time = 0
        self.total_solve_time = 0

        # Loading gene growth rate estimates
        self.g_est = []
        if len(self.config.growth_estimates) > 0:
            print("Loading growth estimates ...")
            for i in range(len(self.config.growth_estimates)):
                self.g_est.append(np.load(self.config.growth_estimates[i]))
        else:
            for i in range(len(self.config.days)):
                self.g_est.append(np.ones(self.train_loader.dataset.rna_sample_num[i]))


        # print(self.config.growth_estimates)
        # print(self.g_est)
        print("ot_growth_iters")
        print(self.config.ot_growth_iters)
        # OT configurations
        self.ot_config = {
            "growth_iters": self.config.ot_growth_iters,
            "epsilon": 0.05,
            "lambda1": 1,
            "lambda2": 50,
            "epsilon0": 1,
            "tau": 1000,
            "scaling_iter": 3000,
            "inner_iter_max": 50,
            "tolerance": 1e-8,
            "max_iter": 1e7,
            "batch_size": 5,
            "extra_iter": 1000,
            "numItermax": 1000000,
            "use_Py": False,
            "use_C": True,
            "profiling": False
        }  # specified in tutorial
        self.ot_solver = compute_transport_map  # ot function

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.config.lr * (0.5 ** ((epoch - 0) // self.config.lr_decay_epoch))
        if (epoch - 0) % self.config.lr_decay_epoch == 0:
            print("LR is set to {}".format(lr))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def train(self, epoch):
        self.model_common_encoder.train()
        self.model_resid_encoder.train()
        self.model_decoder.train()
        self.adjust_learning_rate(self.optimizer, epoch)
        if len(self.config.modalities) > 1:  # multimodal case
            # For multimodal training, use pretrained RNA model as anchor
            # and freeze it weights in early stages
            if epoch < self.config.anchor_epochs:
                # Fix RNA model.
                print("Freezing rna model")
                freeze_model(self.model_common_encoder.rna_encoder, True)
                freeze_model(self.model_decoder.rna_decoder, True)
            else:
                # Jointly train RNA + ATAC. Unfreeze RNA weights.
                print("Un-freeze rna model")
                freeze_model(self.model_common_encoder.rna_encoder, False)
                freeze_model(self.model_decoder.rna_decoder, False)

        # initialize iterator
        total_restored_loss = 0.0
        total_common_loss = 0.0
        total_rna_loss = 0.0
        total_atac_loss = 0.0
        total_alignment_loss = 0.0

        
        time_0 = datetime.now().strftime('%H:%M:%S')
        for batch_idx, (rna_data, atac_data, _, _, indices) in enumerate(self.train_loader):
            rna_data = rna_data.cuda().float()
            atac_data = atac_data.cuda().float()
            bsz, days = rna_data.shape[0], rna_data.shape[1]
            # shape (batch size, days, dim)
            rna_data = rna_data.view(bsz * days, -1)
            atac_data = atac_data.view(bsz * days, -1)
            # model forward
            (
                rna_common_embedding,
                atac_common_embedding,
                rna_cm_bn,
                atac_cm_bn,
            ) = self.model_common_encoder(rna_data, atac_data)
            rna_resid_embedding, atac_resid_embedding, _, _ = self.model_resid_encoder(rna_data, atac_data)
            zero_resid_embedding = torch.zeros_like(rna_resid_embedding)
            rna_restored_rna, atac_restored_atac = self.model_decoder(
                rna_common_embedding,
                atac_common_embedding,
                zero_resid_embedding,
                zero_resid_embedding,
            )

            rna_loss = self.recon_loss(rna_cm_bn, rna_restored_rna)
            atac_loss = self.recon_loss(atac_cm_bn, atac_restored_atac)

            restored_loss = 0
            if "rna" in self.config.modalities:
                restored_loss += rna_loss
            if "atac" in self.config.modalities:
                restored_loss += atac_loss
            restored_loss = restored_loss * self.config.restored_loss_weight

            common_embedding_loss = (
                self.config.common_loss_weight
                * self.criterion_pair(rna_common_embedding, atac_common_embedding).sum()
                / (rna_common_embedding.size()[0] * rna_common_embedding.size()[1])
            )

            total_loss = restored_loss
            if len(self.config.modalities) > 1:
                # If more than 1 modality, then add common embedding loss
                total_loss += common_embedding_loss

            # update encoding weights
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()          

            # OT step
            # pretrain rna reconstruction first than OT = 10
            alignment_loss = 0
            if len(self.rna_gammas) > 0 and self.config.ot_weight > 0:
                (
                    rna_common_embedding,
                    atac_common_embedding,
                    rna_cm_bn,
                    atac_cm_bn,
                ) = self.model_common_encoder(rna_data, atac_data)
                rna_common_embedding = rna_common_embedding.view(bsz, days, -1)
                atac_common_embedding = atac_common_embedding.view(bsz, days, -1)
                # Compute optimal transport loss on rna and atac features
                if "rna" in self.config.modalities:
                    rna_alignment_loss = get_total_ot_loss(rna_common_embedding, indices, self.rna_gammas)
                    alignment_loss += rna_alignment_loss
                
                if "atac" in self.config.modalities:
                    atac_alignment_loss = get_total_ot_loss(atac_common_embedding, indices, self.atac_gammas)
                    alignment_loss += atac_alignment_loss

                alignment_loss *= self.config.ot_weight
                self.optimizer.zero_grad()
                alignment_loss.backward()
                self.optimizer.step()
            else:
                alignment_loss = torch.tensor(0)
            

            # print log
            total_restored_loss += restored_loss.data.item()
            total_common_loss += common_embedding_loss.data.item()
            total_rna_loss += rna_loss.item()
            total_atac_loss += atac_loss.item()
            total_alignment_loss += alignment_loss.item()

            print(
                f"Epoch {epoch+1:2d} [{batch_idx+1:2d}/{len(self.train_loader):2d}] | "
                + f"Recon: {total_restored_loss / (batch_idx + 1):.3f} |"
                + f"RNA: {total_rna_loss / (batch_idx + 1):.3f} |"
                + f"ATAC: {total_atac_loss / (batch_idx + 1):.3f} |"
                + f"pair: {total_common_loss / (batch_idx + 1):.3f} |"
                + f"ot: {total_alignment_loss / (batch_idx + 1):.8f} | "
            )
        # """
        time_2 = datetime.now().strftime('%H:%M:%S')
        ae_time = datetime.strptime(time_2, '%H:%M:%S') - datetime.strptime(time_0, '%H:%M:%S')
        if self.total_ae_time == 0:
            self.total_ae_time = ae_time
        else:
            self.total_ae_time += ae_time
        print(f"AE time: {ae_time} | Total AE Time: {self.total_ae_time}")
        # """
        # Update transport matrices
        if (epoch + 1) % self.config.ot_epochs == 0:
            rna_feats_all, atac_feats_all = get_all_features(self.model_common_encoder, self.test_loaders)
            time_3 = datetime.now().strftime('%H:%M:%S')
            if "rna" in self.config.modalities:
                print("Solving RNA OT")
                solve_ot(
                    rna_feats_all,
                    self.ot_solver,
                    self.ot_config,
                    self.rna_gammas,
                    self.config.days,
                    self.g_est,
                )


            if "atac" in self.config.modalities:
                print("Solving ATAC OT")
                solve_ot(
                    atac_feats_all,
                    self.ot_solver,
                    self.ot_config,
                    self.atac_gammas,
                    self.config.days,
                    self.g_est,
                )
            # """
            time_4 = datetime.now().strftime('%H:%M:%S')

            acc_time = datetime.strptime(time_3, '%H:%M:%S') - datetime.strptime(time_2, '%H:%M:%S')
            solve_time = datetime.strptime(time_4, '%H:%M:%S') - datetime.strptime(time_3, '%H:%M:%S')
            if self.total_accum_time == 0:
                self.total_accum_time = acc_time
                self.total_solve_time = solve_time
            else:
                self.total_accum_time += acc_time
                self.total_solve_time += solve_time

            print(f"Accum time: {acc_time} | Total Accum: {self.total_accum_time}")
            print(f"Solve time: {solve_time} | Total Solve: {self.total_solve_time}")
            # """

        # save checkpoint
        if (epoch + 1) % self.config.save_epochs == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_decoding_state_dict": self.model_decoder.state_dict(),
                    "model_encoding_state_dict": self.model_common_encoder.state_dict()  # ,
                    # 'model_encoding_resid_state_dict': self.model_resid_encoder.state_dict()
                },
                output_dir=self.config.output_dir,
                filename=f"checkpoint_{self.config.exp_name}_e{epoch + 1}.pth.tar",
            )

    def write_embeddings(self, epoch):
        self.model_common_encoder.eval()
        self.model_resid_encoder.eval()
        self.model_decoder.eval()

        for i in range(len(self.test_loaders)):
            print("Writing output of day ", self.config.days[i])
            copyfile(
                "config.py",
                f"{self.config.output_dir}/{self.config.exp_name}/config.py",
            )
            rna_emb_file = f"{self.config.output_dir}/{self.config.exp_name}/rna_embeddings_{i}_{epoch}.npy"
            rna_restored_file = f"{self.config.output_dir}/{self.config.exp_name}/rna_restored_{i}_{epoch}.npy"
            atac_emb_file = f"{self.config.output_dir}/{self.config.exp_name}/atac_embeddings_{i}_{epoch}.npy"
            atac_restored_file = f"{self.config.output_dir}/{self.config.exp_name}/atac_restored_{i}_{epoch}.npy"
            common_emb_file = f"{self.config.output_dir}/{self.config.exp_name}/common_embeddings_{i}_{epoch}.npy"

            rna_bn_mean = self.model_common_encoder.batchnorm.running_mean.data.cpu().numpy()
            rna_bn_std = np.sqrt(self.model_common_encoder.batchnorm.running_var.data.cpu().numpy())

            atac_bn_mean = self.model_common_encoder.atac_batchnorm.running_mean.data.cpu().numpy()
            atac_bn_std = np.sqrt(self.model_common_encoder.atac_batchnorm.running_var.data.cpu().numpy())

            rna_embedding_all = []
            atac_embedding_all = []
            rna_restored_all = []
            atac_restored_all = []

            for batch_idx, (rna_data, atac_data, rna_label, atac_label, _) in enumerate(self.test_loaders[i]):
                rna_data = rna_data.cuda().float()
                atac_data = atac_data.cuda().float()

                # model forward
                (
                    rna_common_embedding,
                    atac_common_embedding,
                    rna_cm_bn,
                    atac_cm_bn,
                ) = self.model_common_encoder(rna_data, atac_data)
                (
                    rna_resid_embedding,
                    atac_resid_embedding,
                    _,
                    _,
                ) = self.model_resid_encoder(rna_data, atac_data)
                # Here! Use zero resid?
                zero_resid_embedding = torch.zeros_like(rna_resid_embedding)
                rna_restored_rna, atac_restored_atac = self.model_decoder(
                    rna_common_embedding,
                    atac_common_embedding,
                    zero_resid_embedding,
                    zero_resid_embedding,
                )
                rna_embedding_all.append(rna_common_embedding.detach().cpu().numpy())
                atac_embedding_all.append(atac_common_embedding.detach().cpu().numpy())
                rna_restored_all.append(rna_restored_rna.detach().cpu().numpy())
                atac_restored_all.append(atac_restored_atac.detach().cpu().numpy())

            rna_embedding_all = np.concatenate(rna_embedding_all, axis=0)
            atac_embedding_all = np.concatenate(atac_embedding_all, axis=0)
            rna_restored_all = np.concatenate(rna_restored_all, axis=0)
            atac_restored_all = np.concatenate(atac_restored_all, axis=0)

            # write raw embedding
            rna_embedding_all = postprocess_feats(rna_embedding_all, normalize=False, rev=False)
            atac_embedding_all = postprocess_feats(atac_embedding_all, normalize=False, rev=False)
            # write restored
            
            if self.config.rna_input == "counts":
              rna_restored_all = postprocess_feats(
                  rna_restored_all,
                  normalize=False,
                  rev=True,
                  bn_mean=rna_bn_mean,
                  bn_std=rna_bn_std,
                  exp=True,
              )
            
            if self.config.rna_input == "logcounts":
              rna_restored_all = postprocess_feats(
                  rna_restored_all,
                  normalize=False,
                  rev=True,
                  bn_mean=rna_bn_mean,
                  bn_std=rna_bn_std,
                  exp=False,
              )
            
            atac_restored_all = postprocess_feats(
                atac_restored_all,
                normalize=False,
                rev=True,
                bn_mean=atac_bn_mean,
                bn_std=atac_bn_std,
                exp=False,
            )
          
            common_embeddings = (rna_embedding_all + atac_embedding_all)/2
            
            np.save(rna_emb_file, rna_embedding_all)
            np.save(rna_restored_file, rna_restored_all)
            np.save(atac_emb_file, atac_embedding_all)
            np.save(atac_restored_file, atac_restored_all)
            np.save(common_emb_file, common_embeddings)
    
    def write_ot(self, epoch):
      days = self.config.days
      common_feats_all = []
      for i in range(len(days)):
        common_feats_all.append(np.load(f"{self.config.output_dir}/{self.config.exp_name}/common_embeddings_{int(i)}_{epoch}.npy"))
        
    
      for d1 in range(len(days)):
        for d2 in range(len(days)):
          if days[d2] <= days[d1]:
            continue
          else:
            print(f"Solving common embedding OT: from day{days[d1]} to day{days[d2]}")
            delta_days = float(days[d2]) - float(days[d1])
            g = np.power(self.g_est[d1], delta_days)
            common_gammas = self.ot_solver(common_feats_all[d1], common_feats_all[d2], self.ot_config, G=g)
            np.save(f"{self.config.output_dir}/{self.config.exp_name}/common_gamma_day{days[d1]}_day{days[d2]}_wad.npy", common_gammas)
            




def postprocess_feats(embedding, normalize=False, rev=False, bn_mean=None, bn_std=None, exp=True):
    if rev:  # reverse batchnorm
        restored_vector = embedding * bn_std + bn_mean
        # reverse log1p
        if exp:
            restored_vector = np.expm1(restored_vector)
    else:
        if normalize:
            embedding = embedding / np.sqrt(np.nansum(np.square(embedding), axis=1, keepdims=True))
        restored_vector = embedding
    return restored_vector
  
  
  
      
      

    

