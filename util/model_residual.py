import torch
import torch.nn as nn


class Net_common_encoder(nn.Module):
    def __init__(self, rna_input_size, atac_input_size, out_dim, hidden_dim=1000):
        super(Net_common_encoder, self).__init__()
        self.rna_input_size = rna_input_size
        self.atac_input_size = atac_input_size

        self.rna_encoder = nn.Sequential(
            nn.Linear(self.rna_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

        self.atac_encoder = nn.Sequential(
            nn.Linear(self.atac_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, rna_data, atac_data):
        rna_embedding_common = self.rna_encoder(rna_data)
        atac_embedding_common = self.atac_encoder(atac_data)
        return rna_embedding_common, atac_embedding_common


class Net_resid_encoder(nn.Module):
    def __init__(self, rna_input_size, atac_input_size, out_dim, hidden_dim=1000):
        super(Net_resid_encoder, self).__init__()
        self.rna_input_size = rna_input_size
        self.atac_input_size = atac_input_size

        self.rna_encoder = nn.Sequential(
            nn.Linear(self.rna_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

        self.atac_encoder = nn.Sequential(
            nn.Linear(self.atac_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, rna_data, atac_data):
        rna_embedding_resid = self.rna_encoder(rna_data)
        atac_embedding_resid = self.atac_encoder(atac_data)

        return rna_embedding_resid, atac_embedding_resid


class Net_concat_decoder(nn.Module):
    def __init__(self, input_dim, rna_output_size, atac_output_size, hidden_dim=1000):
        super(Net_concat_decoder, self).__init__()
        self.rna_output_size = rna_output_size
        self.atac_output_size = atac_output_size

        self.rna_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.rna_output_size),
        )

        self.atac_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.atac_output_size),
        )

    def forward(
        self,
        rna_embedding_common,
        atac_embedding_common,
        rna_embedding_resid,
        atac_embedding_resid,
    ):

        rna_embedding = rna_embedding_common + rna_embedding_resid
        atac_embedding = atac_embedding_common + atac_embedding_resid

        rna_restored_rna = self.rna_decoder(rna_embedding)
        atac_restored_atac = self.atac_decoder(atac_embedding)

        return rna_restored_rna, atac_restored_atac


class BN_common_encoder(nn.Module):
    def __init__(self, rna_input_size, atac_input_size, out_dim, hidden_dim=1000):
        super(BN_common_encoder, self).__init__()
        self.rna_input_size = rna_input_size
        self.atac_input_size = atac_input_size

        self.rna_encoder = nn.Sequential(
            nn.Linear(self.rna_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

        self.atac_encoder = nn.Sequential(
            nn.Linear(self.atac_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

        self.batchnorm = nn.BatchNorm1d(self.rna_input_size, affine=False)
        self.atac_batchnorm = nn.BatchNorm1d(self.atac_input_size, affine=False)

    def forward(self, rna_data, atac_data):
        rna_bn = self.batchnorm(rna_data)
        rna_embedding_common = self.rna_encoder(rna_bn)

        atac_bn = self.atac_batchnorm(atac_data)
        atac_embedding_common = self.atac_encoder(atac_bn)

        return rna_embedding_common, atac_embedding_common, rna_bn, atac_bn


class BN_resid_encoder(nn.Module):
    def __init__(self, rna_input_size, atac_input_size, out_dim, hidden_dim=1000):
        super(BN_resid_encoder, self).__init__()
        self.rna_input_size = rna_input_size
        self.atac_input_size = atac_input_size

        self.rna_encoder = nn.Sequential(
            nn.Linear(self.rna_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

        self.atac_encoder = nn.Sequential(
            nn.Linear(self.atac_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

        self.batchnorm = nn.BatchNorm1d(self.rna_input_size, affine=False)
        self.atac_batchnorm = nn.BatchNorm1d(self.atac_input_size, affine=False)

    def forward(self, rna_data, atac_data):
        rna_bn = self.batchnorm(rna_data)
        rna_embedding_resid = self.rna_encoder(rna_bn)

        atac_bn = self.atac_batchnorm(atac_data)
        atac_embedding_resid = self.atac_encoder(atac_bn)

        return rna_embedding_resid, atac_embedding_resid, rna_bn, atac_bn


class BN_concat_decoder(nn.Module):
    def __init__(self, input_dim, rna_output_size, atac_output_size, hidden_dim=1000):
        super(BN_concat_decoder, self).__init__()
        self.rna_output_size = rna_output_size
        self.atac_output_size = atac_output_size

        self.rna_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.rna_output_size),
            nn.BatchNorm1d(self.rna_output_size),
        )

        self.atac_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.atac_output_size),
        )

    def forward(
        self,
        rna_embedding_common,
        atac_embedding_common,
        rna_embedding_resid,
        atac_embedding_resid,
    ):
        rna_embedding = rna_embedding_common + rna_embedding_resid
        atac_embedding = atac_embedding_common + atac_embedding_resid

        rna_restored_rna = self.rna_decoder(rna_embedding)
        atac_restored_atac = self.atac_decoder(atac_embedding)

        return rna_restored_rna, atac_restored_atac

