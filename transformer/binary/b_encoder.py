import torch
from torch import nn
from torch.nn import Parameter

from transformer.binary.b_attention import MultiHeadedAttention
from transformer.binary.b_quant import QuantizeLinear, gelu
from transformer.binary.config import ModelConfig


class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = QuantizeLinear(config.hidden_size,
                                    config.intermediate_size,
                                    config=config)

    def forward(self, hidden_states, layer_num=0):
        hidden_states = self.dense(hidden_states,
                                   type="layer" + str(layer_num) + "_dense")
        # hidden_states = gelu(hidden_states)
        hidden_states = nn.ReLU()(hidden_states)
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


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadedAttention(config)
        self.intermediate = Intermediate(config)
        self.output = EncodeOutput(config)

    def forward(self, hidden_states, attention_mask, layer_num=0):
        attention_output, layer_att, value_att, context_score, query_score, key_score = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output, layer_num=layer_num)
        layer_output = self.output(intermediate_output, attention_output, layer_num=layer_num)

        return layer_output, layer_att, value_att, context_score, query_score, key_score


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = [hidden_states]
        all_encoder_atts = []
        all_value_atts = []
        all_context_scores = []
        all_query_scores = []
        all_key_scores = []

        for _, layer_module in enumerate(self.layer):
            hidden_states, layer_att, value_att, context_score, query_score, key_score = layer_module(
                hidden_states, attention_mask, layer_num=_)
            all_encoder_layers.append(hidden_states)
            all_encoder_atts.append(layer_att)
            all_value_atts.append(value_att)
            all_context_scores.append(context_score)
            all_query_scores.append(query_score)
            all_key_scores.append(key_score)

        return all_encoder_layers, all_encoder_atts, all_value_atts, all_context_scores, all_query_scores, all_key_scores


if __name__ == '__main__':
    config = ModelConfig(num_attention_heads=4, hidden_size=128,
                         input_bits=1, clip_val=2.5, quantize_act=True, weight_bits=1,
                         attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12,
                         hidden_dropout_prob=0.1, intermediate_size=1024, num_hidden_layers=6)
    model = Encoder(config)
    hidden_state = torch.rand(size=(1, 10, 128))
    hidden_mask = torch.ones(size=(1, 10, 10))
    all_encoder_layers, all_encoder_atts, all_value_atts, all_context_scores, all_query_scores, all_key_scores = model(
        hidden_state, hidden_mask)
