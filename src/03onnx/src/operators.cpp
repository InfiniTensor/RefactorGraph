#include "onnx/operators.h"
#include "computation/operator.h"
#include "infer/infer.h"

namespace refactor::onnx {

    void register_() {
        OpType::register_("onnx::ConstantOfShape", inferConstantOfShape);
        OpType::register_("onnx::Relu", inferUnary);
        OpType::register_("onnx::Sqrt", inferUnary);
        OpType::register_("onnx::Add", inferArithmetic);
        OpType::register_("onnx::Sub", inferArithmetic);
        OpType::register_("onnx::Mul", inferArithmetic);
        OpType::register_("onnx::Div", inferArithmetic);
        OpType::register_("onnx::Pow", inferPow);
        OpType::register_("onnx::MatMul", inferMatMul);
        OpType::register_("onnx::Gemm", inferGemm);
        OpType::register_("onnx::CumSum", inferCumSum);
        OpType::register_("onnx::Max", inferMax);
        OpType::register_("onnx::Transpose", inferTranspose);
        OpType::register_("onnx::Cast", inferCast);
        OpType::register_("onnx::Slice", inferSlice);
        OpType::register_("onnx::Shape", inferShape);
        OpType::register_("onnx::Reshape", inferReshape);
        OpType::register_("onnx::Gather", inferGather);
        OpType::register_("onnx::Concat", inferConcat);
        OpType::register_("onnx::Expand", inferExpand);
        OpType::register_("onnx::Where", inferWhere);
        OpType::register_("onnx::Squeeze", inferSqueeze);
        OpType::register_("onnx::Unsqueeze", inferSqueeze);
        OpType::register_("onnx::Equal", inferEqual);
        OpType::register_("onnx::Softmax", inferSoftmax);
        OpType::register_("onnx::ReduceMean", inferReduce);
    }

}// namespace refactor::onnx
