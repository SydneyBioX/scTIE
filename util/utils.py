import sys
import time
import os
import torch
import shutil
import numpy as np
import random

last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    """Progress Bar for display"""

    def _format_time(seconds):
        days = int(seconds / 3600 / 24)
        seconds = seconds - days * 3600 * 24
        hours = int(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds * 1000)

        f = ""
        i = 1
        if days > 0:
            f += str(days) + "D"
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + "h"
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + "m"
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + "s"
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + "ms"
            i += 1
        if f == "":
            f = "0ms"
        return f

    _, term_width = os.popen("stty size", "r").read().split()
    term_width = int(term_width)
    TOTAL_BAR_LENGTH = 30.0
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    # L.append('     Step: %s' % _format_time(step_time))
    L.append(" | Tot: %s" % _format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2)):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n\n")
    sys.stdout.flush()


# refer to https://github.com/xternalz/WideResNet-pytorch
def save_checkpoint(state, output_dir, filename="checkpoint.pth.tar"):
    """Saves checkpoint to disk"""
    # directory = "/mnt/HDD4/brian_results/models/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(state, os.path.join(output_dir, filename))


def get_all_features(model, loaders):
    print("Accumulating features ...")
    model.eval()
    rna_feats_all = []
    atac_feats_all = []
    for i in range(len(loaders)):
        rna_feats = []
        atac_feats = []
        for batch_idx, (rna_data, atac_data, _, _, _) in enumerate(loaders[i]):  # no shuffle for test loader
            rna_data = rna_data.cuda().float()
            atac_data = atac_data.cuda().float()
            rna_feat, atac_feat, _, _ = model(rna_data, atac_data)
            rna_feats.append(rna_feat.detach().cpu())
            atac_feats.append(atac_feat.detach().cpu())
            del rna_feat
            del atac_feat
            torch.cuda.empty_cache()
        rna_feats = torch.cat(rna_feats, dim=0)  # collect all features
        atac_feats = torch.cat(atac_feats, dim=0)
        rna_feats_all.append(rna_feats)
        atac_feats_all.append(atac_feats)
    return rna_feats_all, atac_feats_all


def freeze_model(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze


def clip_entropy(probs, cell_ids, threshold=1):
    entropy = (-probs * np.log(probs)).sum(axis=1)  # in nat
    ids = np.where(entropy < threshold)[0]  # test = 0.5 (p ~ 0.8)
    return probs[ids], cell_ids[ids]


def clip_high_probs(probs, cell_ids, threshold=0.6):
    max_prob = np.max(probs, axis=1)
    ids = np.where(max_prob > threshold)  # 0.7 ~ 0.26 nat, 0.6 ~ 0.29 nat
    return probs[ids], cell_ids[ids]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
