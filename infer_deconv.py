import os
import numpy as np
import random
import torch
import torch.nn as nn
import time
import datetime
import scipy
import pandas as pd
from config import Config
from util.utils import *
from util.closs import PairLoss, L1regularization
from util.backprop_utils import VanillaBackprop
from util.saliency_utils import read_cell_ids, subsample_cell
from util.dataloader import PrepareUnprocessedDataloaderSampledProbs, UnprocessedDataloaderSampledProbs
from util.model_residual import BN_common_encoder
from argparse import ArgumentParser




if __name__ == "__main__":
    time_start = time.time()
    
    # Set random seed for reproducibility

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser()
    parser.add_argument("--DB", type=str, help = "")
    parser.add_argument("--day1", type=int, nargs="+", 
        help="Starting days (can have multiple)")
    parser.add_argument("--day2", type=int, nargs="+", 
        help="Ends days (can have multiple)")
    parser.add_argument("--day1_cluster", type=str, 
        help="Starting cluster (one)")
    parser.add_argument("--day2_clusters", type=str, nargs="+", 
        help="Ending clusters (can have multiple. 2 in previous experiments.")


    parser.add_argument("--epochs", type=int, default=200, 
        help="Training epochs for classification.")
    parser.add_argument("--lr", type=float, default=0.001, 
        help="Learning rate for lienar classifier.")

    parser.add_argument("--clip_max_probs", type=float, default=0.0, 
        help="Minimum probability (confidence) threshold. Remove all training data under this threshold.")
    parser.add_argument("--common_loss_weight", type=float, default=0.1, 
        help="Weight for RNA-ATAC L2 distance loss.")
    parser.add_argument("--atac_l1_weight", type=float, default=0.001, 
        help="Weight for ATAC encoder L1 reg. loss.")
    parser.add_argument("--rna_l1_weight", type=float, default=0.0, 
        help="Weight for RNA encoder L1 reg. loss.")

    parser.add_argument("--input_dir", type=str, default="", required=True, 
        help="Folder where input data is stored.")
    parser.add_argument("--pretrained_name", type=str, required=True, 
        help="Pretrained model name.")
    parser.add_argument("--pretrained_checkpoints_dir", type=str, default="",
        help="Folder where pretrained model weights are stored.")
    parser.add_argument("--pretrained_results_dir", type=str, default="",
        help="Folder where pretrained model transport matrices are stored.")
    parser.add_argument("--output_dir", type=str, default="./output",
        help="Folder to store trained classifier weights and gradients.")
    parser.add_argument("--seed", type=int, default=1, 
        help="Seed.")
    parser.add_argument("--filter_peaks", type=str, default= None,
        help="Directory of a list of peaks being filtered out")
    args = parser.parse_args()

    rna_all_days_std_dir = os.path.join(args.input_dir, "rna_all_days_std.npy")
    
    if os.path.exists(rna_all_days_std_dir) is not True:
      config = Config(DB = args.DB)
      data = []
      for file_name in config.rna_paths:
          data.append(scipy.sparse.load_npz(file_name))
      data =  scipy.sparse.vstack(data)
      stds = np.std(np.log1p(data * 10000/data.sum(axis = 1)), axis = 0, ddof=1)
      np.save(f"{args.input_dir}/rna_all_days_std.npy", np.squeeze(np.array(stds)))
    
    rna_all_days_std = np.load(rna_all_days_std_dir)
    
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = Config(DB = args.DB, days = args.day1)

    day1_cluster = args.day1_cluster
    day2_clusters = args.day2_clusters
    
    
    d1_str = "-".join([str(d) for d in args.day1])
    d2_str = "-".join([str(d) for d in args.day2])
    name = f"day{d1_str}_{day1_cluster}_to_day{d2_str}_{day2_clusters[0]}_{day2_clusters[1]}_l1reg_rna{args.rna_l1_weight}_atac{args.atac_l1_weight}_pair{args.common_loss_weight}"


    save_dir = os.path.join(args.output_dir, name)
    os.makedirs(save_dir, exist_ok=True)
    

    all_day1_ids = []
    all_day1_probs = []
    total_lib_size = 0
    for d1 in args.day1:
        day1_ids = read_cell_ids(args.input_dir, f"day{d1}_{day1_cluster}.csv")
        all_probs = []
        weights = []
        for d2 in args.day2:
            # Accumulating probabilities
            if d2 <= d1:
                continue
            # print(d1, d2)
            day2_cluster1_ids = read_cell_ids(args.input_dir, f"day{d2}_{day2_clusters[0]}_from_day{d1}_{day1_cluster}.csv")
            day2_cluster2_ids = read_cell_ids(args.input_dir, f"day{d2}_{day2_clusters[1]}_from_day{d1}_{day1_cluster}.csv")
            gamma = np.load(os.path.join(args.pretrained_results_dir, args.pretrained_name, f"common_gamma_day{d1}_day{d2}_wad.npy"))
            lib_size_new = gamma.shape[0]
            gamma = gamma[day1_ids]
            day2_cluster1_prob = gamma[:, day2_cluster1_ids].sum(axis=1, keepdims=True)
            day2_cluster2_prob = gamma[:, day2_cluster2_ids].sum(axis=1, keepdims=True)

            probs = np.concatenate([day2_cluster1_prob, day2_cluster2_prob], axis=1)

            probs = probs / np.sum(probs, axis=1, keepdims=True)
            all_probs.append(probs)
            weights.append(np.sum(probs, axis=1, keepdims=True))
        
        # Rescale probabilities
        weighted_probs = sum(p * w for (p, w) in zip(all_probs, weights))
        all_probs = weighted_probs / sum(weights)
        np.save(os.path.join(save_dir, f"prob_day{d1}_{day1_cluster}.npy"), all_probs)

        all_day1_probs.append(all_probs)

        # Shift the ids of training data by current accumulated library size. 
        # For dataloader sampling.
        day1_ids += total_lib_size
        total_lib_size += lib_size_new
        all_day1_ids.append(day1_ids)
        

    all_day1_probs = np.concatenate(all_day1_probs, axis=0)
    all_day1_ids = np.concatenate(all_day1_ids)
    # print((all_day1_probs))
    # print((all_day1_ids))
    # Remove training data with probabilities under specified threshold.
    all_day1_probs, all_day1_ids = clip_high_probs(all_day1_probs, all_day1_ids, args.clip_max_probs)

    housekeeping_genes_idx = None


    if args.filter_peaks is not None:
      filter_idx = np.genfromtxt(args.filter_peaks, delimiter=',')[1:, 1].astype(int)
      filter_idx.sort()

    

    encoder = BN_common_encoder(config.rna_size, config.atac_size, config.embedding_size, config.hidden_size).cuda()




    print("Loading state dict ...")
    ckpt = torch.load(os.path.join(args.pretrained_checkpoints_dir, f"checkpoint_{args.pretrained_name}_e500.pth.tar"))
    encoder.load_state_dict(ckpt["model_encoding_state_dict"])
        
    encoder = encoder.float()
    encoder.train()

    l = 1


    # KL divergense loss
    criterion = nn.KLDivLoss()
    rna_classifier = nn.Sequential(
        nn.Linear(config.embedding_size, len(args.day2_clusters)),
        nn.Softmax(dim=-1)
    ).cuda()
    atac_classifier = nn.Sequential(
        nn.Linear(config.embedding_size, len(args.day2_clusters)),
        nn.Softmax(dim=-1)
    ).cuda()

    pair_criterion = PairLoss()
    atac_l1_reg_criterion = L1regularization(weight_decay=args.atac_l1_weight)
    rna_l1_reg_criterion = L1regularization(weight_decay=args.rna_l1_weight)

    optimizer = torch.optim.Adam(
        [{'params': rna_classifier.parameters()}, 
        {'params': atac_classifier.parameters()},
        {'params': encoder.parameters()}
        ], 
        lr = args.lr
    )


    trainloader, testloader = PrepareUnprocessedDataloaderSampledProbs(
        config.rna_paths, config.atac_paths, 
        config.rna_labels, config.atac_labels,
        all_day1_ids, all_day1_probs, config.batch_size, 
        housekeeping_genes_idx, config.rna_input
    ).getloader()

    epochs = args.epochs
    for e in range(epochs):
        epoch_rna_loss = 0
        epoch_atac_loss = 0
        epoch_common_loss = 0
        epoch_l1_loss = 0
        cnt = 0
        for i, (rna, atac, probs) in enumerate(trainloader):
            rna, atac, probs = rna.float().cuda(), atac.float().cuda(), probs.float().cuda()
            if args.filter_peaks is not None:
              atac[:,filter_idx] = 0 # make the filter_idx as 0
            rna_feat, atac_feat, _, _ = encoder(rna, atac)
            rna_pred_probs = rna_classifier(rna_feat)
            atac_pred_probs = atac_classifier(atac_feat)

            rna_loss = criterion(torch.log(rna_pred_probs), probs)
            atac_loss = criterion(torch.log(atac_pred_probs), probs)

            common_loss = args.common_loss_weight * pair_criterion(rna_feat, atac_feat).sum() \
                                    / (rna_feat.size()[0] * rna_feat.size()[1])
            atac_l1_loss = atac_l1_reg_criterion(encoder.atac_encoder)
            rna_l1_loss = rna_l1_reg_criterion(encoder.rna_encoder)
            l1_loss = atac_l1_loss + rna_l1_loss
            loss = rna_loss + atac_loss + common_loss + l1_loss
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_rna_loss += rna_loss.item()
            epoch_atac_loss += atac_loss.item()
            epoch_common_loss += common_loss.item()
            epoch_l1_loss += l1_loss.item()
            cnt += 1

        epoch_rna_loss /= cnt
        epoch_atac_loss /= cnt
        epoch_common_loss /= cnt
        epoch_l1_loss /= cnt

        print(f"Epoch {e} | rna loss {epoch_rna_loss} | atac loss {epoch_atac_loss} | common loss {epoch_common_loss} | l1reg {epoch_l1_loss}")


    if epochs > 0:
        print("Saving checkpoints ...")
        save_checkpoint({
            'encoder': encoder.state_dict(),    
            'rna_cls': rna_classifier.state_dict(),
            'atac_cls': atac_classifier.state_dict()#,
        }, save_dir, filename="checkpoint.pth.tar")

    cls_ckpt = torch.load(os.path.join(save_dir, "checkpoint.pth.tar"))
    rna_classifier.load_state_dict(cls_ckpt["rna_cls"])
    atac_classifier.load_state_dict(cls_ckpt["atac_cls"])
    encoder.load_state_dict(cls_ckpt["encoder"])

    rna_classifier.eval()
    atac_classifier.eval()
    encoder.eval()
    
    

   
    # backprop to generate gradients 

    rna_model_layers = [encoder.batchnorm, *encoder.rna_encoder, *rna_classifier]
    rna_model_layers.pop(0) # remove 1st batchnorm layer in backprop model
    rna_model = nn.Sequential(*rna_model_layers).cuda()

    atac_model_layers = [encoder.atac_batchnorm, *encoder.atac_encoder, *atac_classifier]
    atac_model_layers.pop(0) # remove 1st batchnorm layer in backprop model
    atac_model = nn.Sequential(*atac_model_layers).cuda()

    
    # Create backprop wrapper

    bp_rna = VanillaBackprop(rna_model)
    bp_atac = VanillaBackprop(atac_model)

    # To accumulate gradients
    grad_rna0_all = []
    grad_atac0_all = []
    grad_rna1_all = []
    grad_atac1_all = []

    for i, (rna, atac, probs) in enumerate(testloader):
        rna, atac, probs = rna.float().cuda(), atac.float().cuda(), probs.float().cuda()

        rna = encoder.batchnorm(rna)
        atac = encoder.atac_batchnorm(atac)
        for j in range(rna.shape[0]):
            grad_rna0 = bp_rna.generate_gradients(rna[j].unsqueeze(0), 0)
            grad_atac0 = bp_atac.generate_gradients(atac[j].unsqueeze(0), 0)
            grad_rna1 = bp_rna.generate_gradients(rna[j].unsqueeze(0), 1)
            grad_atac1 = bp_atac.generate_gradients(atac[j].unsqueeze(0), 1)
            
            grad_rna0_all.append(grad_rna0)
            grad_atac0_all.append(grad_atac0)
            grad_rna1_all.append(grad_rna1)
            grad_atac1_all.append(grad_atac1)

    grad_rna0_all = np.stack(grad_rna0_all, axis=0)
    grad_atac0_all = np.stack(grad_atac0_all, axis=0)
    grad_rna1_all = np.stack(grad_rna1_all, axis=0)
    grad_atac1_all = np.stack(grad_atac1_all, axis=0)




    grad_rna0_all = grad_rna0_all * (rna_all_days_std[None, ...])
    grad_rna0_all = np.mean(grad_rna0_all, axis=0)
    grad_atac0_all = np.mean(grad_atac0_all, axis=0)

    np.save(os.path.join(save_dir, f"grad_rna.npy"), grad_rna0_all)
    np.save(os.path.join(save_dir, f"grad_atac.npy"), grad_atac0_all)
    
    print("time spent:", datetime.timedelta(seconds=time.time()-time_start))


        
