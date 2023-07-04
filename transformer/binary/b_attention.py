import math

import torch
from torch import nn
from torch.nn import Parameter

from transformer.binary.b_quant import ZMeanBinaryQuantizer, BinaryQuantizer, QuantizeLinear, SymQuantizer
from transformer.binary.config import ModelConfig


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" %
                             (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.quantize_act = config.quantize_act
        self.query = QuantizeLinear(config.hidden_size,
                                    self.all_head_size,
                                    config=config)
        self.key = QuantizeLinear(config.hidden_size,
                                  self.all_head_size,
                                  config=config)
        self.value = QuantizeLinear(config.hidden_size,
                                    self.all_head_size,
                                    config=config)
        if self.quantize_act:
            self.input_bits = config.input_bits
            if self.input_bits == 1:
                self.act_quantizer = BinaryQuantizer
            else:
                self.act_quantizer = SymQuantizer
            self.register_buffer(
                'clip_query', torch.Tensor([-config.clip_val, config.clip_val]))
            self.register_buffer(
                'clip_key', torch.Tensor([-config.clip_val, config.clip_val]))
            self.register_buffer(
                'clip_value', torch.Tensor([-config.clip_val, config.clip_val]))
            self.register_buffer(
                'clip_attn', torch.Tensor([-config.clip_val, config.clip_val]))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        query_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2))
        query_scores = query_scores / math.sqrt(self.attention_head_size)

        key_scores = torch.matmul(key_layer, key_layer.transpose(-1, -2))
        key_scores = key_scores / math.sqrt(self.attention_head_size)

        value_scores = torch.matmul(value_layer, value_layer.transpose(-1, -2))
        value_scores = value_scores / math.sqrt(self.attention_head_size)

        if self.quantize_act and self.input_bits == 1:
            query_layer = self.act_quantizer.apply(query_layer)
            key_layer = self.act_quantizer.apply(key_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.dropout(attention_scores)

        if self.quantize_act and self.input_bits == 1:
            attention_probs = ZMeanBinaryQuantizer.apply(attention_probs)
            value_layer = self.act_quantizer.apply(value_layer)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores, value_scores, 0, query_scores, key_scores


class SelfOutput(nn.Module):
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = QuantizeLinear(config.hidden_size,
                                    config.hidden_size,
                                    config=config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_parameter('gate', Parameter(torch.ones(1).squeeze()))

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, layer_att, value_att, context_score, query_scores, key_scores = self.self(input_tensor, attention_mask.unsqueeze(1))
        attention_output = self.output(self_output, input_tensor)
        return attention_output, layer_att, value_att, context_score, query_scores, key_scores


if __name__ == '__main__':
    config = ModelConfig(num_attention_heads=4, hidden_size=128,
                         input_bits=1, clip_val=2.5, quantize_act=True, weight_bits=1,
                         attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12,
                         hidden_dropout_prob=0.1)
    hidden_state = torch.rand(size=(2, 10, 128))
    hidden_mask = torch.ones(size=(2, 1, 10))
    model = MultiHeadedAttention(config)
    output, attention_scores, value_scores, _, query_scores, key_scores = model(hidden_state, hidden_mask)
    print(output)
