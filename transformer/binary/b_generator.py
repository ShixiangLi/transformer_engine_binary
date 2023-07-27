import torch
from torch import nn
from torch.nn import Parameter

from transformer.binary.b_quant import QuantizeLinear
from transformer.binary.config import ModelConfig


class ModelOutput(nn.Module):
    def __init__(self, config):
        super(ModelOutput, self).__init__()
        self.relu = nn.Sequential(
            nn.Flatten(),
            QuantizeLinear(config.hidden_size * config.time_steps,
                           1,
                           clip_val=config.clip_init_val,
                           weight_bits=config.weight_bits,
                           input_bits=config.input_bits,
                           weight_layerwise=config.weight_layerwise,
                           input_layerwise=config.input_layerwise,
                           weight_quant_method=config.weight_quant_method,
                           input_quant_method=config.input_quant_method,
                           learnable=config.learnable_scaling,
                           symmetric=config.sym_quant_qkvo),
            nn.Tanh()
        )

    def forward(self, hidden_states):
        output = self.relu(hidden_states)

        return output


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
    )
    model = ModelOutput(config)
    f = torch.rand(size=(1, 30, 16))
    output = model(f)
    print(output)
