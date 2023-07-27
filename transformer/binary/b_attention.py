import math

import torch
from torch import nn
from torch.nn import Parameter
import matplotlib.pyplot as plt

from transformer.binary.b_quant import QuantizeLinear, act_quant_fn, AlphaInit, LearnableBias, ZMeanBinaryQuantizer
from transformer.binary.config import ModelConfig


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_bits = config.input_bits
        self.sym_quant_ffn_attn = config.sym_quant_ffn_attn
        self.sym_quant_qkvo = config.sym_quant_qkvo
        self.input_layerwise = config.input_layerwise
        self.input_quant_method = config.input_quant_method
        self.quantize_attention_probs = not config.not_quantize_attention

        self.query = QuantizeLinear(config.hidden_size,
                                    self.all_head_size,
                                    clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits,
                                    input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise,
                                    input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling,
                                    symmetric=config.sym_quant_qkvo)
        self.key = QuantizeLinear(config.hidden_size,
                                  self.all_head_size,
                                  clip_val=config.clip_init_val,
                                  weight_bits=config.weight_bits,
                                  input_bits=config.input_bits,
                                  weight_layerwise=config.weight_layerwise,
                                  input_layerwise=config.input_layerwise,
                                  weight_quant_method=config.weight_quant_method,
                                  input_quant_method=config.input_quant_method,
                                  learnable=config.learnable_scaling,
                                  symmetric=config.sym_quant_qkvo)
        self.value = QuantizeLinear(config.hidden_size,
                                    self.all_head_size,
                                    clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits,
                                    input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise,
                                    input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling,
                                    symmetric=config.sym_quant_qkvo)

        self.move_q = LearnableBias(self.all_head_size)
        self.move_k = LearnableBias(self.all_head_size)
        self.move_v = LearnableBias(self.all_head_size)

        if config.input_quant_method == 'uniform' and config.input_bits < 32:
            self.register_buffer('clip_query', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_key', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_value', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_attn', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            if config.learnable_scaling:
                self.clip_query = nn.Parameter(self.clip_query)
                self.clip_key = nn.Parameter(self.clip_key)
                self.clip_value = nn.Parameter(self.clip_value)
                self.clip_attn = nn.Parameter(self.clip_attn)
        elif (config.input_quant_method == 'elastic' or config.input_quant_method == 'bwn') and config.input_bits < 32:
            self.clip_query = AlphaInit(torch.tensor(1.0))
            self.clip_key = AlphaInit(torch.tensor(1.0))
            self.clip_value = AlphaInit(torch.tensor(1.0))
            self.clip_attn = AlphaInit(torch.tensor(1.0))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, output_att=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if self.input_bits < 32:
            query_layer = self.move_q(mixed_query_layer)
            key_layer = self.move_k(mixed_key_layer)
            value_layer = self.move_v(mixed_value_layer)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.input_bits < 32:
            query_layer = act_quant_fn(query_layer, self.clip_query, self.input_bits,
                                       quant_method=self.input_quant_method,
                                       symmetric=self.sym_quant_qkvo, layerwise=self.input_layerwise)
            key_layer = act_quant_fn(key_layer, self.clip_key, self.input_bits, quant_method=self.input_quant_method,
                                     symmetric=self.sym_quant_qkvo, layerwise=self.input_layerwise)
            value_layer = act_quant_fn(value_layer, self.clip_value, self.input_bits,
                                       quant_method=self.input_quant_method,
                                       symmetric=self.sym_quant_qkvo, layerwise=self.input_layerwise)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if self.input_bits < 32 and self.quantize_attention_probs:
            attention_probs = act_quant_fn(attention_probs,
                                           self.clip_attn,
                                           self.input_bits,
                                           quant_method=self.input_quant_method,
                                           symmetric=self.sym_quant_ffn_attn,
                                           layerwise=self.input_layerwise)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class SelfOutput(nn.Module):
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = QuantizeLinear(config.hidden_size, config.hidden_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()

        self.self = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, layer_att = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, layer_att


if __name__ == '__main__':
    config = ModelConfig(
        num_attention_heads=4,
        hidden_size=16,
        input_bits=1,
        clip_init_val=2.5,
        quantize_act=True,
        weight_bits=1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        intermediate_size=64,
        num_hidden_layers=6,
        time_steps=30,
        max_position_embeddings=30,
        en_dropout=0.1,
        initializer_range=0.02,
        device='cpu',
        weight_layerwise=True,
        input_layerwise=True,
        weight_quant_method='bwn',
        input_quant_method='bwn',
        learnable_scaling=True,
        sym_quant_qkvo=True,
        sym_quant_ffn_attn=True,
        hidden_act='relu'
    )
    hidden_state = torch.rand(size=(1, 30, 16))
    hidden_mask = torch.ones(size=(1, 1, 30))
    model = Attention(config)
    attention_output, layer_att = model(hidden_state, hidden_mask)
    print(attention_output)
