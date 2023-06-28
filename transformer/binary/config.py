class ModelConfig:
    def __init__(self,
                 num_attention_heads=3,
                 hidden_size=15,
                 input_bits=1,
                 clip_val=1,
                 quantize_act=True,
                 weight_bits=1,
                 attention_probs_dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 hidden_dropout_prob=0.1,
                 intermediate_size=1024,
                 num_hidden_layers=6,
                 time_steps=30,
                 max_position_embeddings=30,
                 en_dropout=0.1,
                 initializer_range=0.02,
                 **kwargs):
        super(ModelConfig, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.input_bits = input_bits
        self.clip_val = clip_val
        self.quantize_act = quantize_act
        self.weight_bits = weight_bits
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.time_steps = time_steps
        self.max_position_embeddings = max_position_embeddings
        self.en_dropout = en_dropout
        self.initializer_range = initializer_range
