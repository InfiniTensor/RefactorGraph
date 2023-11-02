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

## 项目结构

### 构建系统

项目构建系统采用 CMake，方便集成一些第三方库。所有第三方库以 git submodule 的形式导入，公共的位于根目录下的 [3rd-party](/3rd-party/) 目录下，另有专用于 python 前端的 pybind11 位于 [src/09python_ffi](/src/09python_ffi/) 目录下。

整个项目的源码以子项目的形式解耦，放在 `src` 的子目录中，每个子项目有自己的 `CMakeLists.txt`，并由根目录的 `CMakeLists.txt` 调用。src 的每个子目录带有一个编号，其中编号大的可以依赖编号小的，方便维护子项目之间的依赖关系，避免循环依赖。当前已有 00-09 共 10 个子项目，它们之间的依赖关系如下图所示：

```plaintext
┌---------┐ ┌--------------┐ ┌-----------┐ ┌----------┐
| 00cmmon |←| 01graph_topo |←| 03runtime |←| 04kernel |
└---------┘ └--------------┘ └-----------┘ └----------┘
    ↑                              |             ↑
    |       ┌---------------┐      |     ┌---------------┐
    └-------| 02mem_manager |←-----┘     | 05computation |
            └---------------┘            └---------------┘
┌--------------------------------------------┐   ↑
| ┌--------------┐ ┌--------┐ ┌------------┐ |   |
| | 09python_ffi |→| 07onnx |→| 06frontend |-|---┘
| └--------------┘ └--------┘ └------------┘ |
|       |     ┌-----------------┐    ↑       |
|       └----→| 08communication |----┘       |
| frontend    └-----------------┘            |
└--------------------------------------------┘
```

所有子项目使用 `PUBLIC` 依赖向下传递自己依赖，并使用 CMake 的 `target_include_directories` 机制使子项目头文件目录随依赖关系传递。

操作 CMake、构建目录和其他项目管理功能的命令和配置封装在根目录下的 `Makefile` 中，现有的命令包括：

- `build`: 默认命令，构建项目。
- `install-python`: 将 Python 前端安装到系统路径。
- `reconfig`: 清除 CMake 缓存，以重新配置 CMake。
- `clean`: 删除构建目录。
- `clean-log`: 清除日志目录。
- `test`: 执行单元测试。
- `format`: 调用格式化工具。

### 第三方依赖版本汇总

- [fmt 10.1.1](https://github.com/fmtlib/fmt/releases/tag/10.1.0)
- [fmtlog v2.2.1](https://github.com/MengRao/fmtlog/releases/tag/v2.2.1)
- [googletest v1.14.0](https://github.com/google/googletest/releases/tag/v1.14.0)
- [backward-cpp v1.6](https://github.com/bombela/backward-cpp/releases/tag/v1.6)
- [result master](https://github.com/oktal/result)
- [abseil-cpp 20230802.0](https://github.com/abseil/abseil-cpp/releases/tag/20230802.0)

### 源码结构

源码的 10 个子项目的简介如下：

| 序号 | 项目 | 说明
|:---:|:----:|:-
|  0  | [`common`](/src/00common/README.md)               | 用于所有子项目的类型和函数定义。
|  1  | [`graph_topo`](/src/01graph_topo/README.md)       | 与元素类型解耦的图拓扑结构表示，包括存储结构和变换算法。
|  2  | [`mem_manager`](/src/02mem_manager/README.md)     | 存储空间管理抽象。
|  3  | [`runtime`](/src/03runtime/README.md)             | 运行时，执行模型推理的图层。
|  4  | [`kernel`](/src/04kernel/README.md)               | 核函数层，包含核函数库以及从核函数图层下降到运行时图层的算法。
|  5  | [`computation`](/src/05computation/README.md)     | 计算图层，包含算子的语义表示定义，以及在算子语义层次进行图变换的规则定义。
|  6  | [`frontend`](/src/06frontend/README.md)           | 前端图层，支持混合存储不同编程框架导入的前端算子，以及基于前端算子的张量形状推导和动态性消解、常量折叠机制。从前端图层下降到计算图层时，图中的动态性（即输出形状的计算还和形状中的变量）必须全部消解。
|  7  | [`onnx`](/src/07onnx/README.md)                   | onnx 前端算子库。
|  8  | [`communication`](/src/08communication/README.md) | 分布式通信算子库。
|  9  | [`python_ffi`](/src/09python_ffi/README.md)       | Python 前端项目。

点击项目名可以跳转到各个项目的文档。
