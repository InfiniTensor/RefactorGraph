# 大模型自定义算子

## RMS Normalization

### Summary

```plaintext
     ___               →   →
y = (x^2 + δ)^(-1/2) * w * x
```

### Attributes

- **epsilon - FLOAT** (default is `1e-5`): 防止除 0 异常的小数字 ε。

### Inputs

2 Inputs:

- **X(heterogeneous) - T**: 来自之前算子的输入数据张量。形状为 `N1 x N2 ... D`，`Nx` 可以为任意维度，将在长度为 `D` 的最后一个维度上标准化。
- **W(heterogeneous) - T**: 权重张量。形状为 `D`，`D` 为 `X` 的最后一个维度的长度。

### Outputs

1 Output:

- **Y(heterogeneous) - T**: 输出张量。形状与 `X` 相同。

## Attention (with Causual Decoder Mask)

### Summary

Multi-head Self Attention 的封装形式，用于 transformer 模型。

支持使用 kv cache，使用条件由输入和属性综合决定。有以下 6 种情况：

| 序号 | 输入数量 | `max_seq_len` | 使用 kv cache | 输出数量 | cache s 维度 | 备注
|:-:|:-:|:-----:|:-------:|:-:|:------------------------:|:-
| 1 | 3 |     0 | none    | 1 | -                        | -
| 2 | 3 | S > 0 | init    | 3 | `S`                      | `assert(S >= seq_len)`
| 3 | 4 |     0 | inplace | 3 | `past_seq_len + seq_len` | `past_seq_len` 必须是常量
| 4 | 4 | S > 0 | inplace | 3 | `S`                      | `assert(S >= past_seq_len + seq_len)`
| 5 | 6 |     0 | copy    | 3 | `past_seq_len + seq_len` | `past_seq_len` 必须是常量
| 6 | 6 | S > 0 | copy    | 3 | `S`                      | `assert(S >= past_seq_len + seq_len)`

### Attributes

- **max_seq_len - INT** (default is `0`): 最大序列长度，用于初始化 kv cache。

### Inputs

- **query(heterogeneous) - T**: 形状为 `N x n_head x seq_len x head_dim`。
- **key(heterogeneous) - T**: 形状为 `N x n_kv_head x seq_len x head_dim`。
- **value(heterogeneous) - T**: 形状为 `N x n_kv_head x seq_len x head_dim`。
- **past_seq_len(optional) -int64**: 要连接的历史序列长度，必须为标量。不使用 kv cache 时留空。
- **k_cache(optional, heterogeneous) -T**: k 缓存的初始值，形状为 `N x n_kv_head x s x head_dim`，`s` 为不小于 `past_seq_len` 的任意值。不使用或不重置 kv cache 时留空。
- **v_cache(optional, heterogeneous) -T**: v 缓存的初始值，形状为 `N x n_kv_head x s x head_dim`，`s` 为不小于 `past_seq_len` 的任意值。不使用或不重置 kv cache 时留空。

### Outputs

- **output(heterogeneous) - T**: 形状与 `query` 相同。
- **k_cache(optional, heterogeneous) - T**: 形状为 `N x n_kv_head x s x head_dim`。`s` 的值根据 `Summary` 的描述计算。
- **v_cache(optional, heterogeneous) - T**: 形状为 `N x n_kv_head x s x head_dim`。`s` 的值根据 `Summary` 的描述计算。
