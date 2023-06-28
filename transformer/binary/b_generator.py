import torch
from torch import nn
from torch.nn import Parameter

from transformer.binary.b_quant import QuantizeLinear, gelu
from transformer.binary.config import ModelConfig


class ModelOutput(nn.Module):
    def __init__(self, config):
        super(ModelOutput, self).__init__()
        self.rul = nn.Sequential(
            nn.Flatten(),
            QuantizeLinear(config.hidden_size * config.time_steps,
                           1,
                           config=config),
            nn.ReLU()
        )
        self.config = config

    def forward(self, hidden_states):
        out = self.rul(hidden_states)

        return out


if __name__ == '__main__':
    config = ModelConfig(num_attention_heads=4, hidden_size=128,
                         input_bits=1, clip_val=2.5, quantize_act=True, weight_bits=1,
                         attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12,
                         hidden_dropout_prob=0.1, intermediate_size=1024, num_hidden_layers=6,
                         time_steps=30)
    model = ModelOutput(config)
    f = torch.rand(size=(10, 30, 128))
    output = model(f)
    print(output)
