from refactor_graph.frontend.modeling import InfiniTensorModel
from refactor_graph.frontend.transformer import Attention, FeedForward, RMSNorm


class DecoderLayer(InfiniTensorModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.attention_layer = self.make_submodel(
            Attention,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            dtype=config.dtype,
            attention_bias=config.attention_bias,
            use_kv_cache=config.use_cache,
        )
        self.input_layernorm = self.make_submodel(
            RMSNorm,
            config.hidden_size,
            config.rms_norm_eps,
            config.dtype,
            model_name="input_layernorm",
        )
        self.post_attention_layernorm = self.make_submodel(
            RMSNorm,
            config.hidden_size,
            config.rms_norm_eps,
            config.dtype,
            model_name="post_attention_layernorm",
        )
        self.feed_forward_layer = self.make_submodel(
            FeedForward, config.hidden_size, config.intermediate_size, config.dtype
        )

    def __call__(
        self, hidden_states, r_embedding_cos, r_embedding_sin, attention_mask=""
    ):
        super().__call__([hidden_states, r_embedding_cos, r_embedding_sin])
        if attention_mask != "":
            self.inputs.append(attention_mask)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention_layer(
            hidden_states, r_embedding_cos, r_embedding_sin, attention_mask
        )
        hidden_states = self.add(hidden_states, residual)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward_layer(hidden_states)
        hidden_states = self.add(hidden_states, residual)

        self.outputs = [hidden_states]
        return hidden_states
