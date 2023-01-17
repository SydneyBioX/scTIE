import os
import numpy as np
import random
import torch
import torch.nn as nn
import time
import datetime
from tqdm import tqdm
from config import Config
from util.dataloader import PrepareUnprocessedDataloader
from util.model_residual import BN_common_encoder
from util.utils import *
from util.backprop_utils import VanillaBackprop
from argparse import ArgumentParser


if __name__ == "__main__":
  
  time_start = time.time()
  


  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  
  parser = ArgumentParser()
  parser.add_argument("--DB", type=str, help="Dataset names", required=True)
  parser.add_argument("--days", nargs='+', type=int, required=True)
  parser.add_argument("--cell_set_idx_path", type = str, nargs='+', default = None)
  parser.add_argument("--cell_set_name", type = str, default = None)
  parser.add_argument("--average", action="store_true", help="whether to store the average only.")
  parser.add_argument("--pretrained_name", type=str, required=True,help="Pretrained model name.")
  parser.add_argument("--pretrained_checkpoints_dir", type=str, help="Folder where pretrained model weights are stored.")
  parser.add_argument("--save_dir", type=str)
  parser.add_argument("--seed", type=int, default=1, help = "Seed")
  args = parser.parse_args()
  
  DB = args.DB
  days = args.days
  pretrained_name = args.pretrained_name
  pretrained_checkpoints_dir = args.pretrained_checkpoints_dir
  save_dir = args.save_dir
  
  # Set random seed for reproducibility
  seed = args.seed
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  
  cell_set_idx_path = args.cell_set_idx_path
  if cell_set_idx_path is not None:
    assert len(cell_set_idx_path) == len(days), "# of days != # of cell_set_idx_path"
  
  
  
  config = Config(DB = DB, days = days)
  jacob_rna_average_all_days  = np.zeros(config.rna_size * config.embedding_size).reshape(config.embedding_size, config.rna_size)
  jacob_atac_average_all_days = np.zeros(config.atac_size * config.embedding_size).reshape(config.embedding_size, config.atac_size)
  num_cells_total = 0
      
      
  for d in range(len(days)):
    day = days[d]
    cell_idx = None
  
    if cell_set_idx_path is not None:
        ids = np.genfromtxt(cell_set_idx_path[d], delimiter=",").astype(np.int64)
        ids.sort()
        cell_idx = ids
      
    print(f"Calculating embedding gradient for Day {day}")
    config = Config(DB = DB, days=[day]) # all days data
    encoder = BN_common_encoder(config.rna_size, config.atac_size, config.embedding_size, config.hidden_size).cuda()
    
    print("Loading state dict ...")
    ckpt = torch.load(
        os.path.join(
            pretrained_checkpoints_dir,
            f"checkpoint_{pretrained_name}.pth.tar",
        )
    )
    encoder.load_state_dict(ckpt["model_encoding_state_dict"])
    
    encoder = encoder.float()
    encoder.eval()  # freeze weights
    
    test_loaders = []
    for i in range(len(config.rna_paths)):
        _, test_loader = PrepareUnprocessedDataloader(
            [config.rna_paths[i]],
            [config.atac_paths[i]],
            [config.rna_labels[i]],
            [config.atac_labels[i]],
            config.batch_size,
            None,
            config.rna_input,
            0, 
        ).getloader()
        test_loaders.append(test_loader)
    
    
    # backprop to generate gradients
    pre_bn = True
    
    rna_model_layers = [encoder.batchnorm, *encoder.rna_encoder]
    if pre_bn:
        rna_model_layers.pop(0)  # remove 1st batchnorm layer in backprop model
    rna_model = nn.Sequential(*rna_model_layers).cuda()
    
    atac_model_layers = [encoder.atac_batchnorm, *encoder.atac_encoder]
    if pre_bn:
        atac_model_layers.pop(0)  # remove 1st batchnorm layer in backprop model
    atac_model = nn.Sequential(*atac_model_layers).cuda()
    
    
    
    bp_rna = VanillaBackprop(rna_model)
    bp_atac = VanillaBackprop(atac_model)
    os.makedirs(os.path.join(save_dir, f"day{day}"), exist_ok=True)
    
    if cell_idx is not None:
      num_cells = len(cell_idx)
      print(f"Number of cells to be calculated: {num_cells}")
    else:
      num_cells = len(test_loaders[0].dataset)
      cell_idx = range(num_cells)
      print(f"Number of cells to be calculated: {num_cells}")
    
    num_cells_total += num_cells
    
    cell_set_name = args.cell_set_name  
    if cell_set_name is None:
      cell_set_name = cell_idx[0]
      
      
    if args.average:
      print("Output the average only.")
      jacob_rna_average  = np.zeros(config.rna_size * config.embedding_size).reshape(config.embedding_size, config.rna_size)
      jacob_atac_average = np.zeros(config.atac_size * config.embedding_size).reshape(config.embedding_size, config.atac_size)
      
    for cell_id in tqdm(cell_idx):
      rna, atac, _, _, _ = test_loaders[0].dataset.__getitem__(cell_id)
      rna, atac = torch.tensor(rna).float().cuda(), torch.tensor(atac).float().cuda()
      rna = rna.unsqueeze(0)
      atac = atac.unsqueeze(0)
      if pre_bn:
          rna = encoder.batchnorm(rna)
          atac = encoder.atac_batchnorm(atac)
    
      jacob_rna = []
      jacob_atac = []
      for l in range(config.embedding_size):
          grad_rna = bp_rna.generate_gradients(rna[0].unsqueeze(0), l)
          grad_atac = bp_atac.generate_gradients(atac[0].unsqueeze(0), l)
          jacob_rna.append(grad_rna)
          jacob_atac.append(grad_atac)
      jacob_rna = np.stack(jacob_rna, axis=0)
      jacob_atac = np.stack(jacob_atac, axis=0)
      
      if args.average:
        jacob_rna_average += jacob_rna
        jacob_atac_average += jacob_atac
      else:
        np.save(os.path.join(save_dir, f"day{day}", f"jacob_rna_cell{cell_id}.npy"), jacob_rna)
        np.save(os.path.join(save_dir, f"day{day}", f"jacob_atac_cell{cell_id}.npy"), jacob_atac)
      # end for loop for cell_idx
    
    if (args.average and len(days) == 1):
      jacob_rna_average = jacob_rna_average/num_cells
      jacob_atac_average = jacob_atac_average/num_cells
      np.save(os.path.join(save_dir, f"day{day}", f"jacob_rna_cellset_{cell_set_name}_average.npy"), jacob_rna_average)
      np.save(os.path.join(save_dir, f"day{day}", f"jacob_atac_cellset_{cell_set_name}_average.npy"), jacob_atac_average)
      
    if (args.average and len(days) > 1):
      jacob_rna_average_all_days += jacob_rna_average
      jacob_atac_average_all_days += jacob_atac_average
  
      
  if (args.average and len(days) > 1):
    jacob_rna_average_all_days = jacob_rna_average_all_days/num_cells_total
    jacob_atac_average_all_days = jacob_atac_average_all_days/num_cells_total
    np.save(os.path.join(save_dir, f"jacob_rna_cellset_{cell_set_name}_average.npy"), jacob_rna_average_all_days)
    np.save(os.path.join(save_dir, f"jacob_atac_cellset_{cell_set_name}_average.npy"), jacob_atac_average_all_days)
    
    
  print("time spent:", datetime.timedelta(seconds=time.time()-time_start))
  
  
  
      
