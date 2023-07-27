import sys

import torch
# from numpy import unicode
from torch import nn
from torch.nn import Parameter

from transformer.binary.b_attention import Attention, LayerNorm
from transformer.binary.b_quant import QuantizeLinear
from transformer.binary.config import ModelConfig


class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = QuantizeLinear(config.hidden_size, config.intermediate_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = torch.nn.functional.relu
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class EncodeOutput(nn.Module):
    def __init__(self, config):
        super(EncodeOutput, self).__init__()
        self.dense = QuantizeLinear(config.intermediate_size,
                                    config.hidden_size,
                                    config=config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_parameter('gate', Parameter(torch.ones(1).squeeze()))

    def forward(self, hidden_states, input_tensor, layer_num=0):
        hidden_states = self.dense(hidden_states, type="layer" + str(layer_num) + "_dense")
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Output(nn.Module):
    def __init__(self, config):
        super(Output, self).__init__()
        self.dense = QuantizeLinear(config.intermediate_size,
                                    config.hidden_size,
                                    clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits,
                                    input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise,
                                    input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling,
                                    symmetric=config.sym_quant_ffn_attn)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):
    def __init__(self, config):
        super(Layer, self).__init__()
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, layer_att = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, layer_att, attention_output


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        all_encoder_atts = []
        all_encoder_atto = []
        for _, layer_module in enumerate(self.layer):
            all_encoder_layers.append(hidden_states)
            hidden_states, layer_att, layer_atto = layer_module(hidden_states, attention_mask)
            all_encoder_atts.append(layer_att)
            all_encoder_atto.append(layer_atto)

        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_atts, all_encoder_atto


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
    model = Encoder(config)
    hidden_state = torch.rand(size=(1, 30, 16))
    hidden_mask = torch.ones(size=(1, 1, 30))
    all_encoder_layers, all_encoder_atts = model(hidden_state, hidden_mask)
    print(len(all_encoder_layers))
    print(len(all_encoder_atts))
