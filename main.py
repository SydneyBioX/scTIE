import os
import torch
from shutil import copyfile
from datetime import datetime
from argparse import ArgumentParser

from config import Config
from util.trainingprocess_bnresidualpretrain import (
    TrainingProcess as TrainingProcess_log1pResPretrain,
)

from util.utils import set_seed


def main():
  
    parser = ArgumentParser()
    
    parser.add_argument("--modalities", type=str, default=["rna"], nargs="*")
    parser.add_argument("--DB", type=str, default=None, help="")
    parser.add_argument("--ckpt", type=str, default=None, help="")
    parser.add_argument("--days", nargs='+', type=int)
    parser.add_argument("--rna_input", type=str, default="counts", help="", choices=["counts", "logcounts"])

    parser.add_argument("--hidden_size", type=int, default=1000, help="")
    parser.add_argument("--embedding_size", type=int, default=64, help="")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--lr", type=float, default=0.1, help="")
    parser.add_argument("--epochs", type=int, default=500, help="")
    parser.add_argument("--ot_epochs", type=int, default=10, help="")
    parser.add_argument("--anchor_epochs", type=int, default=300, help="")
    
    parser.add_argument("--common_loss_weight", type=float, default=1, help="")
    parser.add_argument("--restored_loss_weight", type=float, default=10, help="")
    parser.add_argument("--ot_weight", type=float, default=0.1, help="")
    parser.add_argument("--ot_growth_iters", type=int, default=3, help="")
    
    parser.add_argument("--output_dir", type=str, default="output/", help="")
    parser.add_argument("--seed", type=int, default = 6666, help="")
    parser.add_argument("--ncores", type=int, default = 0, help="")


    # hardware constraint for speed test
    #os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "0"
    # initialization
    args = parser.parse_args()
    config = Config(DB = args.DB, days = args.days, rna_input = args.rna_input,
    ot_epochs = args.ot_epochs, modalities = args.modalities, ckpt = args.ckpt,
    hidden_size = args.hidden_size, embedding_size = args.embedding_size, 
    batch_size = args.batch_size, lr = args.lr, 
    epochs = args.epochs, anchor_epochs = args.anchor_epochs,
    common_loss_weight = args.common_loss_weight, 
    restored_loss_weight = args.restored_loss_weight, 
    ot_weight = args.ot_weight, output_dir = args.output_dir,
    ot_growth_iters = args.ot_growth_iters,
    seed = args.seed, ncores = args.ncores)
    

    
    torch.set_num_threads(config.ncores + 1)
    
    
    # set seed for reproducibility
    if isinstance(config.seed, int):
        print(f"Set seed for reproducibility: {config.seed}")
        set_seed(config.seed)





    # stage1 training
    print("Training start")
    print("Start time: ", datetime.now().strftime("%H:%M:%S"))

    print(config.modalities)
    print("Checkpoint", config.ckpt)
    print("Common loss weight", config.common_loss_weight)
    print("Restored loss weight", config.restored_loss_weight)
    print("OT weight", config.ot_weight)
    print("Number of cores", config.ncores)
    
    model_stage = TrainingProcess_log1pResPretrain(config)
        
    for epoch in range(config.epochs):
        print("Epoch:", epoch + 1)
        model_stage.train(epoch)
        if (epoch + 1) % config.save_epochs == 0:
            print("Write embeddings")
            model_stage.write_embeddings(config.epochs)
            model_stage.write_ot(config.epochs)

    print("Finished: ", datetime.now().strftime("%H:%M:%S"))



if __name__ == "__main__":
    main()
