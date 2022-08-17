# CDNPG
# Copyright 2022-present NAVER Corp.
# BSD 3-clause

"""PyTorch BERT model. """

import logging
import math
import os
import numpy as np
from copy import deepcopy
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers import BertConfig, BertTokenizer, PreTrainedModel, EncoderDecoderConfig, EncoderDecoderModel, get_linear_schedule_with_warmup, top_k_top_p_filtering
from transformers.modeling_bert import BertEmbeddings, BertSelfOutput, BertIntermediate, BertOutput, BertPooler, BertPreTrainedModel, BertOnlyMLMHead

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutput,
)
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer

logger = logging.getLogger(__name__)
 
    
class BertSelfAttention(nn.Module):
    """
    Augmented the self-attention module with granularity (0-1) as input
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        ##################
        self.Wg = nn.Linear(config.hidden_size, 1)
        #################

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states, # [batch_size x seq_len x hidden_size]
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        
        
        ##################################################################
        # Compute Granularity z (0=template, 1=phrase)
        ##################################################################
        batch_size, seq_len, hidden_size = hidden_states.size()
        device = hidden_states.device
        #granularities = F.gumbel_softmax(self.Wg(hidden_states), tau=0.1, hard=True, eps=1e-10)
        #granularities = granularities[:,:,1]
        query_granularities = key_granularities = torch.sigmoid(self.Wg(hidden_states)) 
                                                            #[batch_size x seq_len x 1]
        if encoder_hidden_states is not None: # cross-attention
            key_granularities = torch.sigmoid(self.Wg(encoder_hidden_states))
        ##################################################################
        # 1. resonance penalty: (template words attend to template words)
        #     A = A ⦿ C, 
        #    where C_ij = (1-z_i)*max(0, 1-z_i-z_j) + z_i*min(1, 1-z_i+z_j)
        #  how to explain: 
        #      consider the discrete case: 
        #      when z_i==0, C_ij=max(0, 1-z_j), 
        #             which means template words (z_i==0) tend to attend to template words (z_j==0)
        #      when z_i==1, C_ij=min(1, z_j), 
        #             which means that phrase words (z_i==1) tend to attend to phrase words (z_j==1) 
        #             except those with long range field (the field penalty to be introduced below)
        #  how to compute C in matrix:
        #    Z_I = z, Z_J = z^T, C = (1-Z_I)*max(0, 1-Z_I-Z_J)+z_I*min(1, 1-Z_I+Z_J)
        #################################################################
        Z_I = query_granularities # [B x L_q x 1]
        Z_J = key_granularities.transpose(2, 1) # [B x 1 x L_k]
        #AA = 1-Z_I-Z_J# [B x L_q x L_k]
        #BB = torch.clamp(AA, min=0)
        #print('max(0, 1-Z_I-Z_J)')
        #print(BB.size())
        #CC = 1-Z_I
        #print('1-Z_I')
        #print(CC.size())
        #DD=CC*BB
        #print('(1-Z_I)*max(0, 1-Z_I-Z_J)')
        #print(DD.size())
        resonance_penalties = (1-Z_I)*torch.clamp(1-Z_I-Z_J, min=0)+Z_I*torch.clamp(1-Z_I+Z_J, max=1) # [B x L_q x L_k]
        penalties = resonance_penalties
        
        if encoder_hidden_states is None and seq_len>2:
        #################################################################
        # 2. scope penalty: (phrase words attend to local words)
        # NOTE: this is only applicable for self-attention instead of cross-attention
        #      A = A ⦿ R, 
        #   where R_ij=1 if |i-j|<(L-2)^{1-z_i} +2 else 0   (z_i controls the range of attention)
        #   namely, R_ij=max(0, min(1, (L-2)^{1-z_i}+2-|i-j|)), and L is the max seq length.
        # how to compute R in matrix:
        # R=max(0, min(1, (L-2)^{1-z}+2-|I-J|)) where I=[0,1,...,L-1], J=[0,1,…,L-1]^T
        ################################################################
            I = torch.arange(seq_len, device=device).unsqueeze(1)
            I_J = I-I.T # [L x L]

            scope_penalties = (seq_len-2)**(1-query_granularities)+2-torch.abs(I_J)
            scope_penalties = torch.clamp(scope_penalties, min=1e-32, max=1) # log(1e-32)=-73
            
            penalties = resonance_penalties * scope_penalties
            #penalties = 0.5*resonance_penalties+0.5 * scope_penalties
            #penalties = resonance_penalties
            #penalties = scope_penalties
            
            
        penalties = penalties[:,None,:,:].repeat(1, self.num_attention_heads, 1, 1)
                                    # [B x L x L] -> [B x H x L x L]
        
        
        
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # [batch_size x n_heads x seq_len x seq_len]

        ##################################################################
        attention_scores = attention_scores + torch.log(torch.clamp(penalties, min=1e-32))
        ##################################################################
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
             # [batch_size x n_heads x seq_len x seq_len]

            
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
########################################################################################  
        outputs = (context_layer, attention_probs, query_granularities) if output_attentions else (context_layer,)
      
        return outputs

    
    
class BertAttention(nn.Module):
    """
    The same as the original huggineface implementation
    """
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)#[batch_size x seq_len x hid_size]
       
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
    

class BertLayer(nn.Module):
    """
    The same as the original huggineface implementation
    """
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
                                              # (attn_probs, granularities)

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights
            # (self_attn_probs, self_granularities, cross_attn_probs, cross_granularities)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs  # (layer_output, self_attn_probs, self_granularities, cross_attn_probs, cross_granularities)
        return outputs


class BertEncoder(nn.Module):
    """
    The same as the original huggineface implementation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        ###########################################################
        all_granularities = () if output_attentions else None
        ###########################################################
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            # layer_outputs = (layer_output, self_attn_probs, self_attn_granu, cross_attn_probs, cross_attn_granu)
            hidden_states = layer_outputs[0]
            
##############################################################################
## allow BERT Layer to return cross attention weights by changing the following code （[1]-> [1:]）:
##############################################################################
            if output_attentions:
                assert len(layer_outputs[1:])==2 or len(layer_outputs[1:])==4
                if len(layer_outputs[1:])==2:
                    all_attentions = all_attentions + (layer_outputs[1],)
                    all_granularities = all_granularities + (layer_outputs[2],)
                else: # both self and cross attention weights
                    all_attentions = all_attentions + (layer_outputs[1], layer_outputs[3],)
                    all_granularities = all_granularities + (layer_outputs[2], layer_outputs[4],)
                    
            
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions, all_granularities] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions, granularities = all_granularities,
        )


class BertModel(BertPreTrainedModel):
    """
    ## The same as the original huggineface implementation
    
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        
##### seq_output, pool_output, attention_outputs

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            granularities=encoder_outputs.granularities,
        )


class BertLMHeadModel(BertPreTrainedModel):
    """
    The same as the original huggineface implementation
    """
    def __init__(self, config):
        super().__init__(config)
        assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
    Returns:
    Example::
        >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
        >>> import torch
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        >>> config = BertConfig.from_pretrained("bert-base-cased")
        >>> config.is_decoder = True
        >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config, return_dict=True)
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> prediction_logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutput(
            loss=lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions, granularities = outputs.granularities
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    
class GATransformer(nn.Module):
    '''
    Decomposable Neural Paraphrase Generation 
    (https://arxiv.org/pdf/1906.09741.pdf)
    '''
    def __init__(self, args, base_model_name='bert-base-uncased'):
        super(GATransformer, self).__init__()   
        
        if args.language == 'chinese': base_model_name = 'bert-base-chinese'
            
        self.tokenizer = BertTokenizer.from_pretrained(base_model_name, cache_dir='./cache/')
   
        # encoders
        if args.model_size == 'base': # H = 12, L = 12, D = 768
            self.encoder_config = BertConfig.from_pretrained(base_model_name, cache_dir='./cache/')
        elif args.model_size == 'dnpg-default': 
            self.encoder_config = BertConfig(vocab_size=30522, hidden_size=450, num_hidden_layers=3,
                num_attention_heads=9, intermediate_size=1024)
            
        self.encoder = BertModel(self.encoder_config) # phrase-level encoder
        self.decoder_config = deepcopy(self.encoder_config)
        self.decoder_config.is_decoder=True
        self.decoder = BertLMHeadModel(self.decoder_config)  # sentence-level decoder
        
        self.config = EncoderDecoderConfig.from_encoder_decoder_configs(self.encoder_config, self.decoder_config)
        self.transformer = EncoderDecoderModel(self.config, self.encoder, self.decoder)
        
    def init_weights(self, m):# Initialize Linear Weight for GAN
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.08, 0.08)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)  
            
    @classmethod       
    def from_pretrained(self, model_dir):
        self.encoder_config = BertConfig.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(path.join(model_dir, 'tokenizer'), do_lower_case=args.do_lower_case)
        self.encoder = BertModel.from_pretrained(path.join(model_dir, 'encoder'))
        self.decoder_config = BertConfig.from_pretrained(model_dir)
        self.decoder = BertForMaskedLM.from_pretrained(path.join(model_dir, 'decoder')) 
           
    def save_pretrained(self, output_dir):   
        def save_module(model, save_path):       
            torch.save(model_to_save.state_dict(), save_path)
        def make_list_dirs(dir_list):
            for dir_ in dir_list: os.makedirs(dir_, exist_ok=True)
        make_list_dirs([path.join(output_dir, name) for name in ['tokenizer', 'encoder', 'decoder']])
        
        model_to_save = self.module if hasattr(self, 'module') else self
        model_to_save.encoder_config.save_pretrained(output_dir) # Save configuration file
        model_to_save.tokenizer.save_pretrained(path.join(output_dir,'tokenizer'))
        model_to_save.encoder.save_pretrained(path.join(output_dir, 'encoder'))
        model_to_save.decoder_config.save_pretrained(output_dir) # Save configuration file
        model_to_save.decoder.save_pretrained(path.join(output_dir, 'decoder'))
    
    def forward(self, context, context_attn_mask, response):
        
        batch_size, max_ctx_len = context.size()
        
        context_hiddens, context_encoding, attn_outputs, granularities = self.encoder(
            context, attention_mask = context_attn_mask, output_attentions=True)  # [batch_size x seq_len x dim]
        
        #### before decoding, collect attention probs and penalties
        # context_hiddens: encoder_hidden states [B x H x L_src x E]   
        # hidden_states: (hids of all, hids of layer1, hids of layer2,..., hids of layer L)
        # attn_outputs: (attn of layer1,..., atten of layer L) 
        #                each is a sinle tuple (self-attn,) with size [B x H x L_tar x L_tar] 
        # granularities: tuple of (granu of layer1,...,granu of layer L)
        #                each is a single tuple (granu,) with size [B x L_src x 1]
        self_attn_probs = attn_outputs[-1] # get the attention weights of the last layer  [B x H x L_tar x L_src]
        self_granularities = granularities[0] # get the granularities of the last layer # [batch_size x seq_len x 1]
        
        
        ## decoding
        dec_input = response[:,:-1].contiguous()
        
        outputs, attn_outputs, granularities = self.decoder(
            dec_input, dec_input.ne(self.tokenizer.pad_token_id).long(), None, None, None, None,
            encoder_hidden_states=context_hiddens, encoder_attention_mask=context_attn_mask,  
            output_attentions=True
        )
        #### collect attention probs and granularities
        # outputs: logits of decoder predictions [B x L_tar x V]   
        # attn_outputs: (self_attn of L1, cross_attn of L1, self_attn of L2, ..., cross_atten of LN) 
        #                with [B x H x L_tar x L_tar] and cross-attn: [B x H x L_tar x L_src]
        # granularities: (self_granu of L1, self_granu of L1, self_granu of L2, ...) with [B x L_tar x 1]
        self_attn_probs, cross_attn_probs = attn_outputs[-2], attn_outputs[-1]
        self_granularities, cross_granularities = granularities[0], granularities[1]
        
        ## calculate     
        batch_size, seq_len, vocab_size = outputs.size()
        dec_target = response[:,1:].clone()
        dec_target[response[:,1:] == self.tokenizer.pad_token_id] = -100
        loss_decoder = CrossEntropyLoss()(outputs.view(-1, vocab_size), dec_target.view(-1)) 
        results = {'loss': loss_decoder}
        
        return results
    
    def validate(self, context, context_attn_mask, response):
        self.eval()
        results = self.forward(context, context_attn_mask, response)
        return results['loss'].item()
    
    def generate(self, input_batch, max_len, beam_size, num_samples=1, mode='beamsearch'):
        self.eval()
        device = next(self.parameters()).device
        context, context_attn_mask = [t.to(device) for t in input_batch[:2]]    
        ground_truth = input_batch[2].numpy()
        
        batch_size, max_ctx_len = context.size()
        
        context_hiddens, context_encoding, attn_outputs, granularities = self.encoder(
            context, attention_mask = context_attn_mask, output_attentions=True)  # [batch_size x seq_len x dim]
        
        predictions = self.transformer.generate( # [(batch_size*num_samples) x seq_len]
            context, attention_mask=context_attn_mask, 
            max_length=max_len, temperature=1.0, 
            num_beams=beam_size, 
            early_stopping=True,
            do_sample = True, 
            pad_token_id = self.tokenizer.pad_token_id,
            bos_token_id = self.tokenizer.cls_token_id, # using BERT tokenizer specifications.
            eos_token_id = self.tokenizer.sep_token_id, 
            decoder_start_token_id = self.tokenizer.cls_token_id,
            num_return_sequences=num_samples
        )
        #print(self.tokenizer.decode(predictions[0], skip_special_tokens=True))
        # to numpy
        sample_words = predictions.data.cpu().numpy()
        sample_lens = np.array([predictions.size(1)])  
        
        context = context.data.cpu().numpy()
        granularities = [g[0,:,0].cpu().numpy() for g in granularities] # [num_layers x seq_len]
        return sample_words, sample_lens, context, ground_truth, granularities
        

