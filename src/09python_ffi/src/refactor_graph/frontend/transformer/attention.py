from refactor_graph.frontend.modeling import InfiniTensorModel, DTYPE
from refactor_graph.frontend.nn import Linear
import numpy as np


class Attention(InfiniTensorModel):
    def __init__(
        self,
        batch_size="batch_size",
        seq_len="seq_len",
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        dtype=DTYPE.F32,
        attention_bias=False,
        use_kv_cache=True,
        past_seq_len="past_seq_len",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.use_kv_cache = use_kv_cache
        self.past_seq_len = past_seq_len
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.total_seq_len = (
            self.seq_len + self.past_seq_len
            if isinstance(self.seq_len, int) and isinstance(self.past_seq_len, int)
            else "total_seq_len"
        )

        self.q_proj = self.make_submodel(
            Linear,
            self.hidden_size,
            self.num_heads * self.head_dim,
            attention_bias,
            self.dtype,
            model_name="q_proj",
        )
        self.k_proj = self.make_submodel(
            Linear,
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            attention_bias,
            self.dtype,
            model_name="k_proj",
        )
        self.v_proj = self.make_submodel(
            Linear,
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            attention_bias,
            self.dtype,
            model_name="v_proj",
        )
        self.o_proj = self.make_submodel(
            Linear,
            self.num_heads * self.head_dim,
            self.hidden_size,
            attention_bias,
            self.dtype,
            model_name="o_proj",
        )
        self.rotary_embedding = self.make_submodel(RotaryEmbedding, self.head_dim)

    def __call__(
        self, hidden_states, r_embedding_cos, r_embedding_sin, attention_mask=""
    ):
        super().__call__([hidden_states, r_embedding_cos, r_embedding_sin])
        if attention_mask != "":
            self.inputs.append(attention_mask)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.reshape(
            query_states,
            self.dynamic_tensor(
                (self.batch_size, self.seq_len, self.num_heads, self.head_dim),
                DTYPE.I64,
            ),
        )
        key_states = self.reshape(
            key_states,
            self.dynamic_tensor(
                (self.batch_size, self.seq_len, self.num_kv_heads, self.head_dim),
                DTYPE.I64,
            ),
        )
        value_states = self.reshape(
            value_states,
            self.dynamic_tensor(
                (self.batch_size, self.seq_len, self.num_kv_heads, self.head_dim),
                DTYPE.I64,
            ),
        )
        query_states = self.transpose(query_states, [0, 2, 1, 3])
        key_states = self.transpose(key_states, [0, 2, 1, 3])
        value_states = self.transpose(value_states, [0, 2, 1, 3])

        query_states = self.rotary_embedding(
            query_states, r_embedding_cos, r_embedding_sin
        )
        key_states = self.rotary_embedding(key_states, r_embedding_cos, r_embedding_sin)

        if self.use_kv_cache:
            key_states, value_states = self.cache_kv(key_states, value_states)

        key_states = self.transpose(key_states, [0, 1, 3, 2])
        if self.num_kv_groups > 1:
            attn_weights = self.matmul_group_k(query_states, key_states)
        else:
            attn_weights = self.matmul(query_states, key_states)

        attn_weights = self.div(
            attn_weights,
            self.sqrt(np.array(self.head_dim).astype(self.dtype.np_type())),
        )

        if "" != attention_mask:
            attn_weights = self.add(attn_weights, attention_mask)

        if self.dtype != DTYPE.F32:
            attn_weights = self.cast(attn_weights, DTYPE.F32)
            attn_weights = self.softmax(attn_weights)
            attn_weights = self.cast(attn_weights, self.dtype)
        else:
            attn_weights = self.softmax(attn_weights)

        if self.num_kv_groups > 1:
            attn_output = self.matmul_group_v(attn_weights, value_states)
        else:
            attn_output = self.matmul(attn_weights, value_states)
        attn_output = self.transpose(attn_output, [0, 2, 1, 3])
        attn_output = self.reshape(
            attn_output,
            self.dynamic_tensor(
                (self.batch_size, self.seq_len, self.num_heads * self.head_dim),
                DTYPE.I64,
            ),
        )
        attn_output = self.o_proj(attn_output)

        self.outputs = [attn_output]
        return attn_output

    def matmul_group_k(self, query_states, key_states):
        key_states = self.unsqueeze(key_states, 2)
        query_states = self.reshape(
            query_states,
            self.dynamic_tensor(
                (
                    self.batch_size,
                    self.num_kv_heads,
                    self.num_kv_groups,
                    self.seq_len,
                    self.head_dim,
                ),
                DTYPE.I64,
            ),
        )
        attn_weights = self.matmul(query_states, key_states)
        attn_weights = self.reshape(
            attn_weights,
            self.dynamic_tensor(
                (self.batch_size, self.num_heads, self.seq_len, self.total_seq_len),
                DTYPE.I64,
            ),
        )
        return attn_weights

    def matmul_group_v(self, attn_weights, value_states):
        key_states = self.unsqueeze(value_states, 2)
        attn_weights = self.reshape(
            attn_weights,
            self.dynamic_tensor(
                (
                    self.batch_size,
                    self.num_kv_heads,
                    self.num_kv_groups,
                    self.seq_len,
                    self.total_seq_len,
                ),
                DTYPE.I64,
            ),
        )
        attn_output = self.matmul(attn_weights, key_states)
        attn_output = self.reshape(
            attn_output,
            self.dynamic_tensor(
                (self.batch_size, self.num_heads, self.seq_len, self.head_dim),
                DTYPE.I64,
            ),
        )
        return attn_output

    def cache_kv(self, key_states, value_states):
        k_cache = f"{self.base_name}/k_cache"
        v_cache = f"{self.base_name}/v_cache"
        key_states = self.concat(
            (k_cache, key_states), axis=2, result=f"{self.base_name}/key_states"
        )
        value_states = self.concat(
            (v_cache, value_states), axis=2, result=f"{self.base_name}/value_states"
        )
        self.init_cache(
            k_cache,
            key_states,
            shape=(
                self.batch_size,
                self.num_kv_heads,
                self.past_seq_len,
                self.head_dim,
            ),
            dtype=self.dtype,
        )
        self.init_cache(
            v_cache,
            value_states,
            shape=(
                self.batch_size,
                self.num_kv_heads,
                self.past_seq_len,
                self.head_dim,
            ),
            dtype=self.dtype,
        )
        return key_states, value_states


class RotaryEmbedding(InfiniTensorModel):
    def __init__(self, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim

    def __call__(self, input, cos, sin):
        """
        Args:
            embedding:(seq_len, head_dim)
            input: (bs, seq_len, num_head, head_dim)
        """
        super().__call__([input, cos, sin])
        embed = self.add(self.mul(input, cos), self.mul(self.rotate_half(input), sin))
        self.outputs = [embed]
        return embed

    def rotate_half(self, x):
        x1 = self.slice(x, -1, 0, self.head_dim // 2)
        x2 = self.slice(x, -1, self.head_dim // 2)
        return self.concat((self.neg(x2), x1), axis=-1)
