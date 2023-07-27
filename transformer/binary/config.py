class ModelConfig:
    def __init__(self,
                 num_attention_heads=4,
                 hidden_size=16,
                 input_bits=1,
                 clip_init_val=2.5,
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
                 device='cpu',
                 weight_layerwise=True,
                 input_layerwise=True,
                 weight_quant_method='bwn',
                 input_quant_method='bwn',
                 learnable_scaling=True,
                 sym_quant_qkvo=True,
                 hidden_act='relu',
                 sym_quant_ffn_attn=True,
                 not_quantize_attention=False,
                 **kwargs
                 ):
        super(ModelConfig, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.input_bits = input_bits
        self.clip_init_val = clip_init_val
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
        self.device = device
        self.weight_layerwise = weight_layerwise
        self.input_layerwise = input_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self.learnable_scaling = learnable_scaling
        self.sym_quant_qkvo = sym_quant_qkvo
        self.hidden_act = hidden_act
        self.sym_quant_ffn_attn = sym_quant_ffn_attn
        self.not_quantize_attention = not_quantize_attention
