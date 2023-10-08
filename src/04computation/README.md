# 计算图

计算图算子设计规则：

1. onnx 支持的算子先与 onnx 保持一致；
2. 前端必定是常量的算子不会下降；

   受影响的算子包括：

   - `onnx::Constant`
   - `onnx::ConstantOfShape`
   - `onnx::Range`
   - `onnx::Shape`

3. 调整前端所有用于形状推导的输入；

   - 也参与计算的改为属性，受影响的算子包括：

     | 前端算子 | 输入序号 | 描述
     |:-------:|:-------:|-
     | `onnx::Reduce` | 1       | 要执行 reduce 的轴
     | `onnx::Slice`  | 1,2,3,4 | 切片的细节描述

   - 不参与计算的删除，受影响的算子包括：

     | 前端算子 | 输入序号 | 描述
     |:-------:|:-------:|-
     | `onnx::Expand`    | 1 | 输出形状
     | `onnx::Reshape`   | 1 | 输出形状
     | `onnx::Split`     | 1 | 每个输出的形状
     | `onnx::Squeeze`   | 1 | 输出形状
     | `onnx::Unsqueeze` | 1 | 输出形状

4. 逻辑类似的前端算子将合并或重命名为更好理解的名字；

   受影响的前端算子包括：

   | 前端算子 | 计算图的映射
   |:-------:|:-:
   | `onnx::Expand`    | `Broadcast`
   | `onnx::Squeeze`   | `Reshape`
   | `onnx::Unsqueeze` | `Reshape`
