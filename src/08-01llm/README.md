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
