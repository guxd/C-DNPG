import random
import numpy as np
import argparse
import json
import tables
import os
import csv
import re
from tqdm import tqdm
import pickle as pkl
from transformers import BertTokenizer

from data_loader import load_dict, save_vecs

class Index(tables.IsDescription):
    pos = tables.Int32Col() # start offset of an utterance
    src_len = tables.Int32Col() # number of tokens from the start of dialog  
    tgt_len = tables.Int32Col() # number of tokens till the end of response
def binarize(pairs, src_tokenizer, tar_tokenizer, output_path):
    """binarize data and save the processed data into a hdf5 file
       :param dialogs: an array of dialogs, 
        each element is a list of <caller, utt, feature> where caller is a string of "A" or "B",
        utt is a sentence, feature is an 2D numpy array 
    """
    f = tables.open_file(output_path, 'w')
    filters = tables.Filters(complib='blosc', complevel=5)
    arr = f.create_earray(f.root, 'sentences', tables.Int32Atom(),shape=(0,),filters=filters)
    indices = f.create_table("/", 'indices', Index, "a table of indices and lengths")
    pos = 0
    
    for i, pair in enumerate(tqdm(pairs)):
        src, tgt = pair
        idx_src = src_tokenizer.encode(src)
        if idx_src[0]!=src_tokenizer.cls_token_id: idx_src = [src_tokenizer.cls_token_id] + idx_src
        if idx_src[-1]!=src_tokenizer.sep_token_id: idx_src = idx_src + [src_tokenizer.sep_token_id]
        arr.append(idx_src)
        
        idx_tgt = tar_tokenizer.encode(tgt)
        if idx_tgt[0]!=tar_tokenizer.cls_token_id: idx_tgt = [tar_tokenizer.cls_token_id] + idx_tgt
        if idx_tgt[-1]!=tar_tokenizer.sep_token_id: idx_tgt = idx_tgt + [tar_tokenizer.sep_token_id]
        arr.append(idx_tgt)
        
        ind = indices.row
        ind['pos'] = pos
        ind['src_len'] = len(idx_src)  
        ind['tgt_len'] = len(idx_tgt)   
        ind.append()
        pos += len(idx_src) + len(idx_tgt)
    f.close()
    


def get_quora_data(data_path):
    """
    https://github.com/dev-chauhan/PQG-pytorch/blob/master/prepro/quora_prepro.py
    """
    pairs = []
    f_pairs = csv.reader(open(data_path, 'r', encoding='utf-8'))
    for row in f_pairs:
        idx, qid1, qid2, q1, q2, is_duplicate = row
        if is_duplicate=='1':
            pairs.append((q1, q2))
    return pairs


def get_twitterurl_data(data_path, task):
    """
    https://languagenet.github.io/
    https://github.com/lanwuwei/Twitter-URL-Corpus
    used in papers:https://arxiv.org/pdf/1909.03588.pdf, https://aclanthology.org/2020.acl-main.535.pdf, https://aclanthology.org/D18-1421.pdf
    """
    pairs = []
    texts = open(data_path, 'r', encoding='utf-8').readlines()
    if task == 'train':
        for line in tqdm(texts):
            if(len(line.split('\t'))!=2): 
                print(f'error@:{ascii(line)}')
                continue
            src, tar = line.split('\t')
            pairs.append((src.strip(), tar.strip()))
    else:
        for line in tqdm(texts):
            if(len(line.split('\t'))!=4): 
                print(f'error@:{ascii(line)}')
                continue
            src, tar, rate, url = line.split('\t')
            if int(rate[1])>3:
                pairs.append((src.strip(), tar.strip()))
    print(len(pairs))
    return pairs
    

def load_data(data_path, data_name):
    data={'train':[],'valid':[], 'test':[]}
    if data_name=='quora':
        question_pairs = get_quora_data(data_path+'questions.csv')
        print("=== #all ===")
        print(len(question_pairs))
        random.shuffle(question_pairs)
        data['train']=question_pairs[:100000]
        data['valid']=question_pairs[-24000:-20000]
        data['test']=question_pairs[-20000:]
        
    elif data_name == 'twitterurl':
        data['train'] = get_twitterurl_data(data_path+'2016_Oct_10--2017_Jan_08_paraphrase.txt', 'train')
        val_data = get_twitterurl_data(data_path+'Twitter_URL_Corpus_test.txt', 'valid') 
        test_data = get_twitterurl_data(data_path+'Twitter_URL_Corpus_train.txt', 'test') 
        data['valid'] = val_data[:1000]
        random.shuffle(test_data)
        data['test'] = test_data[:5000]
        
    return data



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data_set", default='quora', help='quora,  twitterurl')
    parser.add_argument('-m1', "--src_model_name", required=True, default='bert-base-uncased', help='bert-base-uncased, bert-base-chinese, bert-base-german-cased')
    parser.add_argument('-m2', "--tar_model_name", required=True, default='bert-base-uncased', help='bert-base-uncased, bert-base-chinese')
    return parser.parse_args()
 
if __name__ == "__main__":
    args=get_args()
    
    work_dir = "./data/"
    data_dir = work_dir + args.data_set+'/'
    
    print("loading data...")
    data = load_data(data_dir, args.data_set)
        
    train_data=data["train"]
    valid_data=data["valid"]
    test_data=data["test"]    
    
    src_tokenizer = BertTokenizer.from_pretrained(args.src_model_name, do_lower_case=True)
    tar_tokenizer = BertTokenizer.from_pretrained(args.tar_model_name, do_lower_case=True)
    
    print('binarizing training data')
    train_out_path = os.path.join(data_dir, "train.h5")
    train_data_binary=binarize(train_data, src_tokenizer, tar_tokenizer, train_out_path)
    
    print('binarizing validation data')
    dev_out_path = os.path.join(data_dir, "valid.h5")
    dev_data_binary = binarize(valid_data, src_tokenizer, tar_tokenizer, dev_out_path) 
    
    print('binarizing test data')
    test_out_path = os.path.join(data_dir, "test.h5")
    test_data_binary = binarize(test_data, src_tokenizer, tar_tokenizer, test_out_path) 
    
    ### test binarized by visualization
 #   dialog=train_data[0]
 #   for caller, utt, feature in dialog['utts']:
 #       print(caller+':'+utt.lower())
            
    table = tables.open_file(train_out_path)
    data = table.get_node('/sentences')
    index = table.get_node('/indices')
    for offset in range(100):
        pos, src_len, tgt_len = index[offset]['pos'], index[offset]['src_len'], index[offset]['tgt_len']
        print('pos:{}, src_len:{}, trg_len:{}'.format(pos, src_len, tgt_len))
        print('src:'+ src_tokenizer.decode(data[pos:pos+src_len]))
        print('tgt:'+ tar_tokenizer.decode(data[pos+src_len:pos+src_len+tgt_len]))
