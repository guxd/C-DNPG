# CDNPG
# Copyright 2022-present NAVER Corp.
# BSD 3-clause

import os
import random
from copy import deepcopy
import numpy as np
import tables
import json
import itertools
from tqdm import tqdm
import torch
import torch.utils.data as data
import logging
logger = logging.getLogger(__name__)

class TransformerDataset(data.Dataset):
    """
    A base class for Transformer dataset
    """
    def __init__(self, file_path, tokenizer, 
                 max_src_len=20, max_tgt_len=20):
        # 
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        table = tables.open_file(file_path)
        self.data = table.get_node('/sentences')[:].astype(np.long)
        self.index = table.get_node('/indices')[:]
        self.data_len = self.index.shape[0]

    def __getitem__(self, offset):
        pos, src_len, tgt_len = self.index[offset]['pos'], self.index[offset]['src_len'], self.index[offset]['tgt_len']
        
        src_arr=self.data[pos: pos+src_len].tolist()
        tgt_arr=self.data[pos+src_len:pos+src_len+tgt_len].tolist()
               
        src = src_arr[1:]
        src_len = min(len(src),self.max_src_len)
        src = src[:src_len-1] + [self.tokenizer.sep_token_id] 
        
        src_attn_mask=[1]*len(src)
        
        tgt = tgt_arr[1:]              
        tgt_len = min(len(tgt),self.max_tgt_len)
        tgt = tgt[:tgt_len-1] + [self.tokenizer.sep_token_id] 
        
        src = self.list2array(src, self.max_src_len, pad_idx=self.tokenizer.pad_token_id) 
        src_attn_mask = self.list2array(src_attn_mask, self.max_src_len) 
        tgt = self.list2array(tgt, self.max_tgt_len, pad_idx=self.tokenizer.pad_token_id)# for decoder training

        return src, src_attn_mask, tgt

    
    def list2array(self, L, d1_len, d2_len=0, d3_len=0, dtype=np.long, pad_idx=0):
        '''  convert a list to an array or matrix  '''            
        def list_dim(a):
            if type(a)!=list: return 0
            elif len(a)==0: return 1
            else: return list_dim(a[0])+1
        
        if type(L) is not list:
            print("requires a (nested) list as input")
            return None
        
        if list_dim(L)==0: return L
        elif list_dim(L) == 1:
            arr = np.zeros(d1_len, dtype=dtype)+pad_idx
            for i, v in enumerate(L): arr[i] = v
            return arr
        elif list_dim(L) == 2:
            arr = np.zeros((d2_len, d1_len), dtype=dtype)+pad_idx
            for i, row in enumerate(L):
                for j, v in enumerate(row):
                    arr[i][j] = v
            return arr
        elif list_dim(L) == 3:
            arr = np.zeros((d3_len, d2_len, d1_len), dtype=dtype)+pad_idx
            for k, group in enumerate(L):
                for i, row in enumerate(group):
                    for j, v in enumerate(row):
                        arr[k][i][j] = v
            return arr
        else:
            print('error: the list to be converted cannot have a dimenson exceeding 3')

    def __len__(self):
        return self.data_len  
    
 
    
def load_dict(filename):
    return json.loads(open(filename, "r").readline())

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs

def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()
    
