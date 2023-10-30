# 重构图表示

[文档](docs/index.md)

## 安装

使用 `make install-python` 编译并安装 Python 前端到全局，安装的包名为 `refactor_grpah`。

## 使用前端

```python
import sys
from onnx import load
from refactor_graph.onnx import make_compiler

model = load(sys.argv[1]) # ----------------------- 加载 onnx 模型
compiler = make_compiler(model) # ----------------- 生成编译器对象
compiler.substitute("seq_len", 1) # --------------- 代换模型中的变量
executor = compiler.compile("cuda", ["ce", "lp"]) # 编译模型到执行器，传入目标硬件和优化选项
```

## 依赖

- [fmt 10.1.1](https://github.com/fmtlib/fmt/releases/tag/10.1.0)
- [fmtlog v2.2.1](https://github.com/MengRao/fmtlog/releases/tag/v2.2.1)
- [googletest v1.14.0](https://github.com/google/googletest/releases/tag/v1.14.0)
- [backward-cpp v1.6](https://github.com/bombela/backward-cpp/releases/tag/v1.6)
- [result master](https://github.com/oktal/result)
- [abseil-cpp 20230802.0](https://github.com/abseil/abseil-cpp/releases/tag/20230802.0)
