# 重构图表示

[![Build](https://github.com/InfiniTensor/RefactorGraph/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/InfiniTensor/RefactorGraph/actions)
[![issue](https://img.shields.io/github/issues/InfiniTensor/RefactorGraph)](https://github.com/InfiniTensor/RefactorGraph/issues)
![license](https://img.shields.io/github/license/InfiniTensor/RefactorGraph)

## 目录

- [安装](#安装)
- [使用前端](#使用前端)
- [项目结构](#项目结构)
  - [构建系统](#构建系统)
  - [子项目简介](#子项目简介)
  - [第三方依赖](#第三方依赖)
- [技术要点](#技术要点)
  - [图拓扑抽象](#图拓扑抽象)

## 安装

使用 `make install-python` 编译并安装 Python 前端到全局，安装的包名为 `refactor_grpah`。

## 使用前端

```python
import sys
import numpy as np
from onnx import load
from refactor_graph.onnx import make_compiler
from onnxruntime import InferenceSession

model = load(sys.argv[1])  # ------------------------------------ 加载模型
input = np.random.random((10, 3, 224, 224)).astype(np.float32)  # 加载测试样本

compiler = make_compiler(model)  # ------------------------------ 模型导入到编译器
compiler.substitute("N", 10)  # --------------------------------- 代换输入中的变量
executor = compiler.compile("cuda", "default", [])  # ----------- 编译模型（选择平台、分配器和优化选项）
executor.set_input(0, input)  # --------------------------------- 设置输入
executor.prepare()  # ------------------------------------------- 准备推理（分配输出空间）
executor.run()  # ----------------------------------------------- 推理

session = InferenceSession(model.SerializeToString())  # -------- 与 onnxruntime 对比结果以验证推理
answer = session.run(None, {session.get_inputs()[0].name: input})
print([(executor.get_output(i) - answer[i]).flatten() for i in range(len(answer))])
```

对于使用外部数据的模型，支持直接加载以减少一次拷贝：

```python
import sys
from pathlib import Path
from onnx import load
from refactor_graph.onnx import make_compiler

model_path = Path(sys.argv[1])  # ---------------------------- 假设模型和数据保存在相同路径
model = load(model_path.as_uri(), load_external_data=False)  # 不直接加载外部数据以避免额外拷贝
compiler = make_compiler(model, model_path.parent.as_uri())  # 导入时直接加载外部数据
executor = compiler.compile("cuda", "default", [])  # -------- 编译模型

# 下同
```

## 项目结构

### 构建系统

项目构建系统采用 CMake，方便集成一些第三方库。所有第三方库以 git submodule 的形式导入，公共的位于根目录下的 [3rd-party](/3rd-party/) 目录下，另有专用于 python 前端的 pybind11 位于 [src/09python_ffi](/src/09python_ffi/) 目录下。

整个项目的源码以子项目的形式解耦，放在 `src` 的子目录中，每个子项目有自己的 `CMakeLists.txt`，并由根目录的 `CMakeLists.txt` 调用。src 的每个子目录带有一个编号，其中编号大的可以依赖编号小的，方便维护子项目之间的依赖关系，避免循环依赖。当前已有 00-09 共 10 个子项目，它们之间的依赖关系如下图所示：

```plaintext
┌─────────┐ ┌──────────────┐ ┌───────────┐ ┌──────────┐
│ 00cmmon ├←┤ 01graph_topo ├←┤ 03runtime ├←┤ 04kernel │
└───┬─────┘ └──────────────┘ └─────┬─────┘ └─────┬────┘
    ↑                              │             ↑
    │       ┌───────────────┐      │     ┌───────┴───────┐
    └───────┤ 02mem_manager ├←─────┘     │ 05computation │
            └───────────────┘            └───────┬───────┘
┌────────────────────────────────────────────┐   ↑
│ ┌──────────────┐ ┌────────┐ ┌────────────┐ │   │
│ │ 09python_ffi ├→┤ 07onnx ├→┤ 06frontend ├─┼───┘
│ └─────┬────────┘ └────────┘ └──────┬─────┘ │
│       │     ┌─────────────────┐    ↑       │
│       └────→┤ 08communication ├────┘       │
│ frontend    └─────────────────┘            │
└────────────────────────────────────────────┘
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

### 子项目简介

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

### 第三方依赖

- [fmt 10.1.1](https://github.com/fmtlib/fmt/releases/tag/10.1.0)
- [fmtlog v2.2.1](https://github.com/MengRao/fmtlog/releases/tag/v2.2.1)
- [googletest v1.14.0](https://github.com/google/googletest/releases/tag/v1.14.0)
- [backward-cpp v1.6](https://github.com/bombela/backward-cpp/releases/tag/v1.6)
- [result master](https://github.com/oktal/result)
- [abseil-cpp 20230802.0](https://github.com/abseil/abseil-cpp/releases/tag/20230802.0)

## 技术要点

### 多层计算图表示

目前主流的深度学习框架几乎都应用了两种基本的设计，即计算图和多层 IR，本框架也应用了这两种设计。

计算图是 AI 模型计算的一种表示形式。在 AI 模型的高层表示中，通常使用以算子为节点、张量为边的图结构来表示模型，例如，ONNX 模型可视化后通常表现为这样的形式：

![onnx model](/docs/images/README-1.png)

计算图的本质在逻辑上是一张数据流图，在数据结构上是一张有向无环图。数据流图意味着其中的节点表示一种运算的过程，而边表示在数据在运算之间流动的起点和终点。与经典的数据结构意义上的有向无环图相比，计算图中节点的输入和输出是有序的，不可互换，例如某个节点入度为 3 不是对节点输入的完备描述，必须说明第 0 个、第 1 个、第 2 个入边分别是什么。

产品级的 AI 编译器总是具有较长的工作流程。首先，模型从多种框架的高层表示（Pytorch/ONNX/……）转化到编译器定义的计算图形式，然后应用各种图上的变换，在确定的硬件上选择合适的计算方法（kernel），最后实际执行。在流程的不同阶段，需要关注和维护的信息是不同的。举例来说，`Reshape` 算子表示改变张量形状的语义，在高层的模型表示中是必要的，但在 AI 应用程序实际执行时，`Reshape` 算子不会真正改变数据，甚至可以直接从图上删除。因此，最好使用多层 IR 表示模型，在不同的层级改变关注的信息，从而降低每层的复杂度。

### 图拓扑抽象

灵活的多层 IR 表示要求层与层之间尽量使用无关的节点和边类型，以保证每层的自由设计，同时要尽量复用拓扑结构的表示，因为无论在哪一层，图结构的表示和操作方法是类似的，因此，本框架采用了拓扑结构与节点/边信息解耦的实现方式。拓扑结构以一系列模板类型的形式定义在 `graph_topo` 子项目中。这些拓扑结构表示被定义成类似容器类型的形式，只包含关键的节点和边的有向无环、出入有序的基本信息，对节点和边具体是什么没有约束，从而支持在不同的计算图中复用这些容器，并能在不与编译器的业务耦合的情况下编写、优化和测试，具有良好的工程特性。

目前，图拓扑表示包含以下主要的类型定义：

- `GraphTopo`: 精简、连续存储的拓扑类型，用于持久化存储和、遍历和快速随机访问；
- `Searcher`: 基于 `GraphTopo` 引用建立的查询缓存结构，用于包含图上一些冗余但常用的信息，使这些信息不必一直随着拓扑结构移动或拷贝；
- `Builder`: 所有字段都公开可访问的结构体类型，可以表示拓扑结构并支持开发者自由操作，用于快速构建拓扑结构，并提供一个方法将拓扑信息压缩，转化成 `GraphTopo`；
- `LinkedGraph`: 链式的拓扑表示，在这种表示中修改拓扑连接关系、增删节点具有 `O(1)` 时间复杂度，但空间复杂度更高、不保证维持拓扑序，访问时也更容易缓存不命中；

参考子项目依赖关系图，几乎所有依赖 `graph_topo` 的子项目（除了前端算子库和接口层）都定义了一套最适于表示其信息的节点和边定义，并复用上述拓扑结构类型来表示一种特定的计算图。
