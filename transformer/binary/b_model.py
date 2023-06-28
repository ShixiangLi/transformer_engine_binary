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

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids)
        encoded_layers, attention_scores, value_scores, context_scores, query_scores, key_scores = self.encoder(embedding_output, extended_attention_mask)

        output = self.outer(encoded_layers[-1])
        return encoded_layers, attention_scores, output, value_scores, context_scores, query_scores, key_scores
        # return output


if __name__ == '__main__':
    config = ModelConfig()
    model = Model(config)
    hidden_state = torch.rand(size=(10196, 30, 15))
    hidden_mask = torch.ones(size=(10196, 1, 30))
    encoded_layers, attention_scores, output, value_scores, context_scores, query_scores, key_scores = model(
        hidden_state, hidden_mask)
