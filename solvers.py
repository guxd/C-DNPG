# C-DNPG
# Copyright 2022-present NAVER Corp.
# BSD 3-clause

import os
import logging
import torch
#try:
#    from torch.utils.tensorboard import SummaryWriter
#except:
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm, trange

from model import GATransformer
from data_loader import TransformerDataset
from learner import Learner

logger = logging.getLogger(__name__)

    
def get_optim_params(models, args):
    no_decay = ['bias', 'LayerNorm.weight']
    parameters = []
    for model in models:
        parameters.append(
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': args.weight_decay})
        parameters.append(
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0})
    return parameters

class Solver(object):
    def __init__(self, args, model=None):
        self.model = model    
        self.build(args)
        
    def build(self, args):
        # Load pretrained model and tokenizer
        raise NotImplementedError
        
    def load(self, args):
        # Load a trained model and vocabulary that you have fine-tuned
        raise NotImplementedError
        
    def train(self, args):     
        if args.local_rank not in [-1, 0]: torch.distributed.barrier()# only the first process process the dataset, others use cache     
        train_set = TransformerDataset(os.path.join(args.data_path, 'train.h5'), self.model.tokenizer, max_src_len=args.src_maxlen, max_tgt_len=args.tar_maxlen)
        valid_set = TransformerDataset(os.path.join(args.data_path, 'valid.h5'), self.model.tokenizer, max_src_len=args.src_maxlen, max_tgt_len=args.tar_maxlen)
        test_set = TransformerDataset(os.path.join(args.data_path, 'test.h5'), self.model.tokenizer, max_src_len=args.src_maxlen, max_tgt_len=args.tar_maxlen)
        if args.local_rank == 0: torch.distributed.barrier() # end of barrier
        
        optim_params = get_optim_params([self.model], args)
        global_step, tr_loss = Learner().run_train(
            args, self.model, train_set, optim_params, entry='forward', max_steps = args.max_steps, valid_set=valid_set, test_set=test_set)
        
        return global_step, tr_loss
    
    def evaluate(self, args):
        eval_output_dir = f"./output/{args.model}/"
        if args.local_rank in [-1, 0]: 
            os.makedirs(eval_output_dir, exist_ok=True)
            
        test_set = TransformerDataset(os.path.join(args.data_path, 'test.h5'), self.model.tokenizer, max_src_len=args.src_maxlen, max_tgt_len=args.tar_maxlen)
        best_valid_loss, result, generated_text = Learner().run_eval(
            args, self.model, test_set
        )
        
        with open(os.path.join(eval_output_dir, f"eval_results.html"), 'w') as f_eval:
            f_eval.write(generated_text+'\n')
        return result
   
class GATransformerSolver(Solver):
    def __init__(self, args, model=None):
        super(GATransformerSolver, self).__init__(args, model)
        
    def build(self, args):
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()# ensure only the first process in distributed training download model&vocab
        
        if self.model is None:
            self.model = GATransformer(args)    
        self.model.to(args.device)

        if args.local_rank == 0:
            torch.distributed.barrier()#End of barrier to ensure only the first process in distributed training download model&vocab
        
    def load(self, args):
        # Load a trained model and vocabulary that you have fine-tuned
        output_dir = os.path.join(f"./output/{args.model}/models/", 'optimal') 
        self.model = GATransformer.from_pretrained(output_dir)
        self.model.to(args.device)
        
        
    def evaluate(self, args):
        eval_output_dir = f"./output/{args.model}/"
        if args.local_rank in [-1, 0]: 
            os.makedirs(eval_output_dir, exist_ok=True)
            
        test_set = TransformerDataset(os.path.join(args.data_path, 'test.h5'), self.model.tokenizer, max_src_len=args.src_maxlen, max_tgt_len=args.tar_maxlen)
        
        best_valid_loss, result, generated_text = Learner().run_eval(args, self.model, test_set)
        with open(os.path.join(eval_output_dir, f"eval_results.html"), 'w') as f_eval:
            f_eval.write(generated_text+'\n')

        
        return result
    
   