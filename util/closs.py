import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def _one_hot(tensor, num):
    b = list(tensor.size())[0]
    onehot = torch.cuda.FloatTensor(b, num).fill_(0)
    ones = torch.cuda.FloatTensor(b, num).fill_(1)
    out = onehot.scatter_(1, torch.unsqueeze(tensor, 1), ones)
    return out


class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.0
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay

        return regularization_loss


def cor(m):
    m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def reduction_loss(embedding, identity_matrix, size):
    loss = torch.mean(torch.abs(torch.triu(cor(embedding), diagonal=1)))
    loss = loss + 1 / torch.mean(
        torch.abs(embedding - torch.mean(embedding, dim=0).view(1, size).repeat(embedding.size()[0], 1))
    )
    loss = loss + torch.mean(torch.abs(embedding))
    return loss


def cosine_sim(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = torch.matmul(x, torch.transpose(y, 0, 1))

    return sim


def cosine_sim_linear_product(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)

    sim = torch.mean(torch.sum(x * y, dim=1))

    return sim


def cosine_sim_linear_product_orthogonal(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)

    sim = torch.mean(torch.abs(torch.sum(x * y, dim=1)))

    return sim


class EncodingLoss(nn.Module):
    def __init__(self, dim=32, p=1):
        super(EncodingLoss, self).__init__()
        self.identity_matrix = torch.tensor(np.identity(dim)).float().cuda()
        self.p = p
        self.dim = dim

    def forward(self, atac_embedding, rna_embedding):
        # rna
        rna_reduction_loss = reduction_loss(rna_embedding, self.identity_matrix, self.dim)
        # atac
        atac_reduction_loss = reduction_loss(atac_embedding, self.identity_matrix, self.dim)

        loss = rna_reduction_loss + atac_reduction_loss
        return loss


class CellLoss(nn.Module):
    def __init__(self):
        super(CellLoss, self).__init__()

    def forward(self, rna_cell_out, rna_cell_label):
        rna_cell_loss = F.cross_entropy(rna_cell_out, rna_cell_label.long())
        return rna_cell_loss


class PairLoss(nn.Module):
    def __init__(self):
        super(PairLoss, self).__init__()

    def forward(self, embedding1, embedding2):
        # pair_loss = F.l1_loss(embedding1.float(), embedding2.float())
        pair_loss = F.mse_loss(embedding1.float(), embedding2.float(), reduction="none")
        # pair_loss = 1 - torch.mean(F.cosine_similarity(embedding1, embedding2))
        return pair_loss


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, embedding1, embedding2):
        loss = nn.KLDivLoss(reduction="batchmean")(embedding1.float(), embedding2.float())
        return loss


class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, embedding1, embedding2):
        loss = nn.NLLLoss()(embedding1.float(), embedding2.float())
        return loss


class PairLossL1(nn.Module):
    def __init__(self):
        super(PairLossL1, self).__init__()

    def forward(self, embedding1, embedding2):
        pair_loss = F.l1_loss(embedding1.float(), embedding2.float())
        return pair_loss


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, embedding1, embedding2, embedding1_resid, embedding2_resid):

        """
        cosine_loss = cosine_sim_linear_product_orthogonal(embedding1.float(), embedding1_resid.float())\
					+ cosine_sim_linear_product_orthogonal(embedding2.float(), embedding2_resid.float())\
					- cosine_sim_linear_product(embedding1.float(), embedding2.float())
        """
        cosine_sim = nn.CosineSimilarity()
        cosine_loss = torch.mean(
            torch.abs(
                cosine_sim(embedding1.float(), embedding1_resid.float())
                + cosine_sim(embedding2.float(), embedding2_resid.float())
            )
        )
        return cosine_loss


class CorrelationLoss(nn.Module):
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, emb1, emb2):

        # corr_loss = - rna_correlation(emb1.float(), emb2.float()) \
        corr_loss = -sample_correlation(emb1.float(), emb2.float())

        return corr_loss


def sample_correlation(emb1, emb2):
    """
    emb size: batch_size x feature_size (ex: 512 x 1965)
    """
    emb1_norm = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb1_norm2 = emb1_norm / torch.linalg.norm(emb1_norm, dim=0, keepdim=True)

    emb2_norm = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb2_norm2 = emb2_norm / torch.linalg.norm(emb2_norm, dim=0, keepdim=True)

    sample_corr = torch.mean(torch.sum(torch.mul(emb1_norm2, emb2_norm2), dim=0))

    return sample_corr


def rna_correlation(emb1, emb2):
    """
    emb size: batch_size x feature_size (ex: 512 x 1965)
    """
    emb1_norm = emb1 - torch.mean(emb1, dim=1, keepdim=True)
    emb1_norm2 = emb1_norm / torch.linalg.norm(emb1_norm, dim=1, keepdim=True)

    emb2_norm = emb2 - torch.mean(emb2, dim=1, keepdim=True)
    emb2_norm2 = emb2_norm / torch.linalg.norm(emb2_norm, dim=1, keepdim=True)

    rna_corr = torch.mean(torch.sum(torch.mul(emb1_norm2, emb2_norm2), dim=1))

    return rna_corr


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, p):
        h = (-p * torch.log(p + 1e-8)).sum(dim=-1)
        h = h.mean()
        return h


class ClusterHeadLoss(nn.Module):
    def __init__(self, margin, num_clusters):
        super(ClusterHeadLoss, self).__init__()
        self.margin = margin
        self.num_clusters = num_clusters
        self.softmax = nn.Softmax(-1)

    def forward(self, model):

        c = torch.tensor(np.arange(self.num_clusters)).cuda()
        c = _one_hot(c, self.num_clusters).float()
        c = model.cluster_head(c)
        x = c[:, 0]
        y = c[:, 1]
        x = x - x.unsqueeze(-1)
        y = y - y.unsqueeze(-1)
        d = torch.sqrt(x ** 2 + y ** 2)
        # d = torch.clip(d, 0, self.margin)
        return -d.mean()


# class KLDivLoss(nn.Module):
#     def __init__(self):
#         super(KLDivLoss, self).__init__()

#     def forward(self, mu, log_var):
#         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
#         return kld_loss
