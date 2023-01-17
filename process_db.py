import numpy as np
import scipy.sparse
import h5py
import sys
import os





def h5_reader(file_name):
    h5 = h5py.File(file_name, "r")
    h5_data = h5["matrix/data"]
    print("H5 dataset shape:", h5_data.shape)

    return h5_data


def to_sparse_mat(file_name):
    h5_data = h5_reader(file_name)
    sparse_data = scipy.sparse.csr_matrix(np.array(h5_data).transpose())
    scipy.sparse.save_npz(file_name.replace("h5", "npz"), sparse_data)



if __name__ == "__main__":
  

  days = [2, 3, 4, 7]
  data_dir = "data/NIPS/"
  for d in days:
    print(d)
    rna_file_name = f"{data_dir}/nips_day{d}_rna.h5"
    to_sparse_mat(rna_file_name)
    atac_file_name = f"{data_dir}/nips_day{d}_atac.h5"
    to_sparse_mat(atac_file_name)
