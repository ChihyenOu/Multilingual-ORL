from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
from torch.nn.functional import embedding_bag, linear
from torch.nn import functional as F
# from transformers.modeling_bert import BertModel
# from driver.modeling import BertModel as BERTModel
from transformers.models.bert.modeling_bert import *


def set_requires_grad(module: nn.Module, status: bool = False):
    for param in module.parameters():
        param.requires_grad = status


def batch_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ batched linear forward """
    y = torch.einsum("bth,boh->bto", x, w)
    y = y + b.unsqueeze(1)
    return y


class Adapter(nn.Module):
    def __init__(self, in_features, bottleneck_size, external_param=False):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_size = bottleneck_size
        self.act_fn = nn.GELU()

        if external_param:
            self.params = [None, None, None, None]
        else:
            self.params = nn.ParameterList([
                nn.Parameter(torch.Tensor(bottleneck_size, in_features)),
                nn.Parameter(torch.zeros(bottleneck_size)),
                nn.Parameter(torch.Tensor(in_features, bottleneck_size)),
                nn.Parameter(torch.zeros(in_features))
            ])
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.params[0], std=1e-3)
        nn.init.normal_(self.params[2], std=1e-3)

    def forward(self, hidden_states: torch.Tensor):
        linear_forward = batch_linear if self.params[0].dim() == 3 else linear
        x = linear_forward(hidden_states, self.params[0], self.params[1])
        x = self.act_fn(x)
        x = linear_forward(x, self.params[2], self.params[3])
        x = x + hidden_states
        return x

class AdapterWithParameterGen(nn.Module):
    def __init__(self, hidden_size, language_emb_size, adapter_size, config, lang_emb = None): # ADD config
        super(AdapterWithParameterGen, self).__init__()
        self.config = config # ADD
        self.lang_emb = lang_emb # ADD
        # Original
        low_rank_dim = language_emb_size
        
        self.down_W = nn.Parameter(torch.zeros(low_rank_dim, hidden_size, adapter_size))
        # print('DOWN:', self.down_W.shape)
        self.down_b = nn.Parameter(torch.zeros(low_rank_dim, adapter_size))

        # self.activation = ACT2FN[config.adapter_act] if isinstance(config.adapter_act, str) else config.adapter_act
        self.activation = nn.GELU()

        self.up_W = nn.Parameter(torch.zeros(low_rank_dim, adapter_size, hidden_size))
        self.up_b = nn.Parameter(torch.zeros(low_rank_dim, hidden_size))
        # Add
        self.low_rank_dim = language_emb_size
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size

        self.init_weights()

    def forward(self, hidden_states, lang_emb=None):
        # Origin: AttributeError: 'AdapterWithParameterGen' object has no attribute 'config'
        # down_w = torch.matmul(lang_emb, self.down_W.view(self.config.low_rank_dim, -1)).view(self.config.hidden_size, self.config.adapter_size)
        # New: #reshape down_W into 2 dims -> do dot product -> reshape down_W into 3 dims
        down_w = torch.matmul(self.lang_emb, self.down_W.view(self.low_rank_dim, -1)).view(self.hidden_size, self.adapter_size) #CHANGE: lang_emb to self.lang_emb
        down_b = torch.matmul(self.lang_emb, self.down_b) #CHANGE: lang_emb to self.lang_emb
        down_projected = F.linear(hidden_states, down_w.t(), down_b)

        activated = self.activation(down_projected)

        # Origin: AttributeError: 'AdapterWithParameterGen' object has no attribute 'config'
        # up_w = torch.matmul(lang_emb, self.up_W.view(self.config.low_rank_dim, -1)).view(self.config.adapter_size, self.config.hidden_size)
        # New
        up_w = torch.matmul(self.lang_emb, self.up_W.view(self.low_rank_dim, -1)).view(self.adapter_size, self.hidden_size)
        up_b = torch.matmul(self.lang_emb, self.up_b)
        up_projected = F.linear(activated, up_w.t(), up_b)

        return hidden_states + up_projected

    def init_weights(self):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        self.down_W.data.normal_(mean=0.0, std=0.0001)
        self.up_W.data.normal_(mean=0.0, std=0.0001)

class AdapterBertOutput(nn.Module):
    """
    替代BertOutput和BertSelfOutput
    """
    def __init__(self, base, adapter_forward):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter_forward

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class AdapterPGNBertOutput(nn.Module):
    """
    替代BertOutput和BertSelfOutput
    """
    def __init__(self, base, adapter_forward): # layer.output, self.adapters[i][0].forward
        super().__init__()
        self.base = base
        self.adapter_forward = adapter_forward

    def forward(self, hidden_states, input_tensor, lang_emb=None):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states, lang_emb) # ADD lang_emb
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class AdapterBertModel(nn.Module):
    def __init__(self,
                 name_or_path_or_model: Union[str, BertModel],
                 adapter_size: int = 128,
                 external_param: Union[bool, List[bool]] = False,
                 word_piece: str = 'first',  # 需要保证input ids为第一个
                 **kwargs):
        super().__init__()
        if isinstance(name_or_path_or_model, str):
            self.bert = BertModel.from_pretrained(name_or_path_or_model)
        else:
            self.bert = name_or_path_or_model

        set_requires_grad(self.bert, False)

        if isinstance(external_param, bool):
            param_place = [external_param for _ in range(
                self.bert.config.num_hidden_layers)]
        elif isinstance(external_param, list):
            param_place = [False for _ in range(
                self.bert.config.num_hidden_layers)]
            for i, e in enumerate(external_param, 1):
                param_place[-i] = e
        else:
            raise ValueError("wrong type of external_param!")

        self.adapters = nn.ModuleList([nn.ModuleList([
                Adapter(self.bert.config.hidden_size, adapter_size, e),
                Adapter(self.bert.config.hidden_size, adapter_size, e)
            ]) for e in param_place
        ])


        for i, layer in enumerate(self.bert.encoder.layer):
            layer.output = AdapterBertOutput(
                layer.output, self.adapters[i][0].forward)
            set_requires_grad(layer.output.base.LayerNorm, True)
            layer.attention.output = AdapterBertOutput(
                layer.attention.output, self.adapters[i][1].forward)
            set_requires_grad(layer.attention.output.base.LayerNorm, True)

        self.output_dim = self.bert.config.hidden_size

        if word_piece == 'first':
            self.word_piece = None
        else:  # mean of pieces
            offset = torch.tensor([0], dtype=torch.long)
            self.word_piece = lambda x: embedding_bag(
                x, self.bert.embeddings.word_embeddings.weight, offset.to(x.device))

    def forward(self,
                input_ids: torch.Tensor,
                mask: torch.Tensor = None,
                word_pieces: Dict[Tuple[int], torch.LongTensor] = None,
                **kwargs) -> torch.Tensor:
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        if self.word_piece is not None and word_pieces is not None:
            for (s, w), pieces in word_pieces.items():
                inputs_embeds[s, w, :] = self.word_piece(pieces)

        attention_mask = None if mask is None else mask.float()
        bert_output = self.bert(attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        return bert_output[0]

class AdapterPGNBertModel(nn.Module):
    def __init__(self,
                 name_or_path_or_model: Union[str, BertModel],
                 adapter_size: int = 128,
                 language_emb_size: int = 32,
                 external_param: Union[bool, List[bool]] = False,
                 config: object = None, # ADD
                 # lang_embedding: object = None # ADD
                 ):
        super().__init__()
        
        # ADD
        self.config = config

        if isinstance(name_or_path_or_model, str):
            # OLD
            self.bert = BertModel.from_pretrained(name_or_path_or_model)
            # New
            # Because: BERTModel.from_pretrained(config.bert_path, config=bert_config)
            # self.bert = BertModel.from_pretrained(name_or_path_or_model, config=config, local_files_only=True) # add config
            # self.bert = BertModel.from_pretrained(name_or_path_or_model, config=config, local_files_only=True) # add config
        else:
            self.bert = name_or_path_or_model

        # ADD
        # self.bert.encoder = BertEncoder(config) AttributeError: 'BertConfig' object has no attribute 'chunk_size_feed_forward'

        set_requires_grad(self.bert, False)

        if isinstance(external_param, bool):
            param_place = [external_param for _ in range(
                self.bert.config.num_hidden_layers)]
        elif isinstance(external_param, list):
            param_place = [False for _ in range(
                self.bert.config.num_hidden_layers)]
            for i, e in enumerate(external_param, 1):
                param_place[-i] = e
        else:
            raise ValueError("wrong type of external_param!")

        print(self.bert.config.hidden_size, language_emb_size, adapter_size)
        
        # self.adapters = 

        self.adapters = nn.ModuleList([nn.ModuleList([
                AdapterWithParameterGen(self.bert.config.hidden_size, language_emb_size, adapter_size, self.config, lang_emb = lang_embedding), # ADD self.config
                AdapterWithParameterGen(self.bert.config.hidden_size, language_emb_size, adapter_size, self.config, lang_emb = lang_embedding) # ADD self.config
            ]) for e in param_place
        ])

        print('Adaptor: ', self.adapters)

        # Original version    
        for i, layer in enumerate(self.bert.encoder.layer):
            layer.output = AdapterPGNBertOutput(    # change to AdapterPGNBertOutput? #original code here use AdapterBertOutput
                layer.output, self.adapters[i][0].forward)    
            set_requires_grad(layer.output.base.LayerNorm, True)
            layer.attention.output = AdapterPGNBertOutput(     # change to AdapterPGNBertOutput? #original code here use AdapterBertOutput
                layer.attention.output, self.adapters[i][1].forward)
            set_requires_grad(layer.attention.output.base.LayerNorm, True)

        # New Version
        """
        for i, layer in enumerate(self.bert.encoder.layer):
            layer.output = AdapterPGNBertOutput(    # change to AdapterPGNBertOutput? #original code here use AdapterBertOutput
                layer.output, self.adapters[i][0](lang_emb = lang_embedding))    
            set_requires_grad(layer.output.base.LayerNorm, True)
            layer.attention.output = AdapterPGNBertOutput(     # change to AdapterPGNBertOutput? #original code here use AdapterBertOutput
                layer.attention.output, self.adapters[i][1](lang_emb = lang_embedding))
            set_requires_grad(layer.attention.output.base.LayerNorm, True)"""

        self.output_dim = self.bert.config.hidden_size

        # if word_piece == 'first':
        #     self.word_piece = None
        # else:  # mean of pieces
        #     offset = torch.tensor([0], dtype=torch.long)
        #     self.word_piece = lambda x: embedding_bag(
        #         x, self.bert.embeddings.word_embeddings.weight, offset.to(x.device))
        
    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.LongTensor = None,
                mask: torch.Tensor = None,
                bert_pieces: torch.LongTensor = None,
                lang_embedding = None # ADD
                ) -> torch.Tensor:
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        
        # if self.word_piece is not None and word_pieces is not None:
        #     for (s, w), pieces in word_pieces.items():
        #         inputs_embeds[s, w, :] = self.word_piece(pieces)

        attention_mask = None if mask is None else mask.float()
        bert_output = self.bert(attention_mask=attention_mask, inputs_embeds=inputs_embeds, token_type_ids=token_type_ids) # , lang_embedding = lang_embedding
        """
        TypeError: BertModel.forward() got an unexpected keyword argument 'lang_embedding'
        def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        """



        # print(len(bert_output))
        # print(self.layer)
        output = torch.bmm(bert_pieces, bert_output[self.layer - 1])
        # return bert_output[0]
        return output

class PGNmodel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.adaptorpgn =  AdapterWithParameterGen(self.bert.config.hidden_size, language_emb_size, adapter_size, self.config)

    def forward(self, hidden_states, input_tensor, lang_emb=None):