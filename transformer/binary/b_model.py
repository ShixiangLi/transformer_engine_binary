import torch
from torch import nn

from transformer.binary.b_embedding import PositionalEncoding
from transformer.binary.b_encoder import Encoder
from transformer.binary.b_generator import ModelOutput
from transformer.binary.config import ModelConfig


class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Model(PreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.embeddings = PositionalEncoding(config)
        self.encoder = Encoder(config)
        self.outer = ModelOutput(config)
        self.apply(self.init_weights)

    def forward(self, input_ids, attention_mask=None, output_all_encoded_layers=True, output_att=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids)
        encoded_layers, layer_atts, layer_atto = self.encoder(embedding_output, extended_attention_mask)

        output = self.outer(encoded_layers[-1])
        return encoded_layers, layer_atts, output, layer_atto


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
    model = Model(config)
    hidden_state = torch.rand(size=(1, 30, 16))
    hidden_mask = torch.ones(size=(1, 1, 30))
    encoded_layers, layer_atts, output = model(hidden_state, hidden_mask)
    print(output)
