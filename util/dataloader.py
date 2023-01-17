import torch
import torch.utils.data as data
import numpy as np
import os
import os.path
import random
import csv
import scipy.sparse

from config import Config


def sample_data_unprocessed(rna_data, atac_data, rna_labels, atac_labels, index):
    rna_sample = rna_data[index].reshape(-1)
    # preprocessed
    rna_sample = rna_sample * 10000 / np.sum(rna_sample)
    rna_sample = np.log1p(rna_sample)
    int_rna_data = (rna_sample).astype(np.float)
    atac_sample = atac_data[index].reshape(-1)
    int_atac_data = (atac_sample > 0).astype(np.float)  # binarize data
    if rna_labels is not None:
        in_rna_label = rna_labels[index]
    else:
        in_rna_label = -1
    if atac_labels is not None:
        in_atac_label = atac_labels[index]
    else:
        in_atac_label = -1
    return int_rna_data, int_atac_data, in_rna_label, in_atac_label, index


def sample_data_processed(rna_data, atac_data, rna_labels, atac_labels, index):
    rna_sample = rna_data[index].reshape(-1)
    # preprocessed
    # rna_sample = rna_sample * 10000 / np.sum(rna_sample)
    # rna_sample = np.log1p(rna_sample)
    int_rna_data = (rna_sample).astype(np.float)
    atac_sample = atac_data[index].reshape(-1)
    int_atac_data = (atac_sample > 0).astype(np.float)  # binarize data
    if rna_labels is not None:
        in_rna_label = rna_labels[index]
    else:
        in_rna_label = -1
    if atac_labels is not None:
        in_atac_label = atac_labels[index]
    else:
        in_atac_label = -1
    return int_rna_data, int_atac_data, in_rna_label, in_atac_label, index


def sparse_mat_reader(file_name):
    data = scipy.sparse.load_npz(file_name)
    print("Read db:", file_name, " shape:", data.shape)
    return data, data.shape[1], data.shape[0]


def load_labels(
    label_file,
):  # please run parsing_label.py first to get the numerical label file (.txt)
    return np.loadtxt(label_file)



def read_multi_file(data_paths, label_paths=None):
    input_sizes, sample_nums = 0, 0
    data_list, label_list = [], []
    for data_path, label_path in zip(data_paths, label_paths):
        data_path = os.path.join(os.path.realpath("."), data_path)
        data, labels = None, None

        data, input_size, sample_num = sparse_mat_reader(data_path)
        input_sizes = input_size
        sample_nums += sample_num
        data_list.append(np.array(data.todense()))

        if label_path is not None:
            label_path = os.path.join(os.path.realpath("."), label_path)
            label = load_labels(label_path)
            label_list.append(label)

    datas = np.vstack(data_list)
    if label_list != []:
        labels = np.vstack(label_list)
    else:
        labels = None

    return input_sizes, sample_nums, datas, labels

# FOR OT
def read_multi_file_list(data_paths, label_paths=None):
    input_sizes, sample_nums = 0, []
    data_list, label_list = [], []
    for data_path, label_path in zip(data_paths, label_paths):
        data_path = os.path.join(os.path.realpath("."), data_path)
        data, labels = None, None

        data, input_size, sample_num = sparse_mat_reader(data_path)
        input_sizes = input_size
        sample_nums.append(sample_num)
        data_list.append(np.array(data.todense()))

        if label_path is not None:
            label_path = os.path.join(os.path.realpath("."), label_path)
            label = load_labels(label_path)
            label_list.append(label)
        else:
            label_list.append(None)
    return input_sizes, sample_nums, data_list, label_list


class PrepareUnprocessedDataloader:
    def __init__(self, rna_path, atac_path, rna_label, atac_label, batch_size, hk_indices, rna_input, ncores):
        # hardware constraint
        kwargs = {"num_workers": ncores, "pin_memory": True}

        # load RNA
        trainset = UnprocessedDataloader(rna_path, atac_path, rna_label, atac_label, hk_indices, rna_input)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)

        trainset = UnprocessedDataloader(rna_path, atac_path, rna_label, atac_label, hk_indices, rna_input)
        self.testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, **kwargs)

    def getloader(self):
        return self.trainloader, self.testloader



class UnprocessedDataloader(data.Dataset):
    def __init__(self, rna_path, atac_path, rna_label_path, atac_label_path, hk_indices, rna_input):
        (
            self.rna_input_size,
            self.rna_sample_num,
            self.rna_data,
            self.rna_labels,
        ) = read_multi_file(rna_path, rna_label_path)
        (
            self.atac_input_size,
            self.atac_sample_num,
            self.atac_data,
            self.atac_labels,
        ) = read_multi_file(atac_path, atac_label_path)
        self.hk_indices = hk_indices
        self.rna_input = rna_input
        if self.hk_indices is not None:
            self.rna_input_size -= len(self.hk_indices)
            total_genes = np.arange(self.rna_data.shape[-1])
            selected_genes = np.delete(total_genes, self.hk_indices)
            self.rna_data = self.rna_data[:, selected_genes]
        print(self.rna_input_size, self.rna_data[0].shape)
        assert self.rna_sample_num == self.atac_sample_num, "# of atac != # of rna"

    def __getitem__(self, index):
      if self.rna_input == "counts":
        return sample_data_unprocessed(self.rna_data, self.atac_data, self.rna_labels, self.atac_labels, index)
      if self.rna_input == "logcounts":
        return sample_data_processed(self.rna_data, self.atac_data, self.rna_labels, self.atac_labels, index)
    def __len__(self):
        return self.rna_sample_num




class UnprocessedDataloaderList(data.Dataset):
    def __init__(self, rna_path, atac_path, rna_label_path, atac_label_path, hk_indices, rna_input):
        (
            self.rna_input_size,
            self.rna_sample_num,
            self.rna_data,
            self.rna_labels,
        ) = read_multi_file_list(rna_path, rna_label_path)
        (
            self.atac_input_size,
            self.atac_sample_num,
            self.atac_data,
            self.atac_labels,
        ) = read_multi_file_list(atac_path, atac_label_path)
        assert self.rna_sample_num == self.atac_sample_num, "# of atac != # of rna"
        self.num_libraries = len(rna_path)
        # print(len(self))
        self.hk_indices = hk_indices
        self.rna_input = rna_input
        if self.hk_indices is not None:
            self.rna_input_size -= len(self.hk_indices)
            for i in range(len(self.rna_data)):
                total_genes = np.arange(self.rna_data[i].shape[-1])
                selected_genes = np.delete(total_genes, self.hk_indices)
                self.rna_data[i] = self.rna_data[i][:, selected_genes]
        print(self.rna_input_size, self.rna_data[0].shape)

    def __getitem__(self, index):
        rna_datas, atac_datas, rna_labels, atac_labels = [], [], [], []
        indices = []
        for i in range(self.num_libraries):
          if self.rna_input == "counts":
            rna_data, atac_data, rna_label, atac_label, index = sample_data_unprocessed(
                self.rna_data[i],
                self.atac_data[i],
                self.rna_labels[i],
                self.atac_labels[i],
                index % self.rna_sample_num[i],
            )
          if self.rna_input == "logcounts":
            rna_data, atac_data, rna_label, atac_label, index = sample_data_processed(
                self.rna_data[i],
                self.atac_data[i],
                self.rna_labels[i],
                self.atac_labels[i],
                index % self.rna_sample_num[i],
            )
          rna_datas.append(rna_data)
          atac_datas.append(atac_data)
          rna_labels.append(rna_label)
          atac_labels.append(atac_label)
          indices.append(index)
        rna_datas = np.stack(rna_datas)
        atac_datas = np.stack(atac_datas)
        rna_labels = np.stack(rna_labels)
        atac_labels = np.stack(atac_labels)
        indices = np.stack(indices)
        return rna_datas, atac_datas, rna_labels, atac_labels, indices

    def __len__(self):
        return max(self.rna_sample_num)



class UnprocessedDataloaderSampledProbs(data.Dataset):
    def __init__(
        self,
        rna_path,
        atac_path,
        rna_label_path,
        atac_label_path,
        indices,
        probs,
        hk_indices,
        rna_input
    ):
        (
            self.rna_input_size,
            self.rna_sample_num,
            self.rna_data,
            self.rna_labels,
        ) = read_multi_file(rna_path, rna_label_path)
        (
            self.atac_input_size,
            self.atac_sample_num,
            self.atac_data,
            self.atac_labels,
        ) = read_multi_file(atac_path, atac_label_path)
        assert self.rna_sample_num == self.atac_sample_num, "# of atac != # of rna"
        assert len(indices) == probs.shape[0]
        self.indices = indices
        self.probs = probs
        self.rna_data = self.rna_data[self.indices, :]
        self.atac_data = self.atac_data[self.indices, :]

        # Modify here!
        self.hk_indices = hk_indices
        self.rna_input = rna_input
        if self.hk_indices is not None:
            self.rna_input_size -= len(self.hk_indices)
            total_genes = np.arange(self.rna_data.shape[-1])
            selected_genes = np.delete(total_genes, self.hk_indices)
            self.rna_data = self.rna_data[:, selected_genes]
        print(self.rna_input_size, self.rna_data[0].shape)

    def __getitem__(self, index):
      if self.rna_input == "counts":
        rna, atac, rna_l, atac_l, ind = sample_data_unprocessed(
            self.rna_data, self.atac_data, self.rna_labels, self.atac_labels, index
        )
        return rna, atac, self.probs[index]
      if self.rna_input == "logcounts":
        rna, atac, rna_l, atac_l, ind = sample_data_processed(
            self.rna_data, self.atac_data, self.rna_labels, self.atac_labels, index
        )
        return rna, atac, self.probs[index]

    def __len__(self):
        return len(self.indices)


class PrepareUnprocessedDataloaderList:
    def __init__(self, rna_path, atac_path, rna_label, atac_label, batch_size, hk_indices, rna_input, ncores):
        # hardware constraint
        kwargs = {"num_workers": ncores, "pin_memory": True}

        # load RNA
        trainset = UnprocessedDataloaderList(rna_path, atac_path, rna_label, atac_label, hk_indices, rna_input)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)

        trainset = UnprocessedDataloaderList(rna_path, atac_path, rna_label, atac_label, hk_indices, rna_input)
        self.testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, **kwargs)

    def getloader(self):
        return self.trainloader, self.testloader


class PrepareUnprocessedDataloaderSampledProbs:
    def __init__(
        self,
        rna_path,
        atac_path,
        rna_label,
        atac_label,
        indices,
        probs,
        batch_size,
        hk_indices,
        rna_input,
    ):
        # hardware constraint
        kwargs = {"num_workers": 0, "pin_memory": True}

        # load RNA
        trainset = UnprocessedDataloaderSampledProbs(
            rna_path, atac_path, rna_label, atac_label, indices, probs, hk_indices,rna_input
        )
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)

        trainset = UnprocessedDataloaderSampledProbs(
            rna_path, atac_path, rna_label, atac_label, indices, probs, hk_indices, rna_input
        )
        self.testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, **kwargs)

    def getloader(self):
        return self.trainloader, self.testloader

