#include "onnx/operators.h"
#include "operators/batch_normalization.hh"
#include "operators/cast.hh"
#include "operators/compair.hh"
#include "operators/concat.hh"
#include "operators/constant.hh"
#include "operators/constant_of_shape.hh"
#include "operators/conv.hh"
#include "operators/cum_sum.hh"
#include "operators/expand.hh"
#include "operators/gather.hh"
#include "operators/gather_elements.hh"
#include "operators/gemm.hh"
#include "operators/global_pool.hh"
#include "operators/mat_mul.hh"
#include "operators/pool.hh"
#include "operators/simple_binary.hh"

namespace refactor::onnx {

    void register_() {
        // clang-format off
        Operator::register_<BatchNormalization>("onnx::BatchNormalization");
        Operator::register_<Cast              >("onnx::Cast"              );
        Operator::register_<Compair           >("onnx::Equal"             );
        Operator::register_<Compair           >("onnx::Greater"           );
        Operator::register_<Compair           >("onnx::GreaterOrEqual"    );
        Operator::register_<Compair           >("onnx::Less"              );
        Operator::register_<Compair           >("onnx::LessOrEqual"       );
        Operator::register_<Concat            >("onnx::Concat"            );
        Operator::register_<Constant          >("onnx::Constant"          );
        Operator::register_<ConstantOfShape   >("onnx::ConstantOfShape"   );
        Operator::register_<Conv              >("onnx::Conv"              );
        Operator::register_<CumSum            >("onnx::CumSum"            );
        Operator::register_<Expand            >("onnx::Expand"            );
        Operator::register_<Gather            >("onnx::Gather"            );
        Operator::register_<GatherElements    >("onnx::GatherElements"    );
        Operator::register_<Gemm              >("onnx::Gemm"              );
        Operator::register_<GlobalPool        >("onnx::GlobalAveragePool" );
        Operator::register_<GlobalPool        >("onnx::GlobalLpPool"      );
        Operator::register_<GlobalPool        >("onnx::GlobalMaxPool"     );
        Operator::register_<MatMul            >("onnx::MatMul"            );
        Operator::register_<Pool              >("onnx::AveragePool"       );
        Operator::register_<Pool              >("onnx::LpPool"            );
        Operator::register_<Pool              >("onnx::MaxPool"           );
        Operator::register_<SimpleBinary      >("onnx::Add"               );
        Operator::register_<SimpleBinary      >("onnx::Sub"               );
        Operator::register_<SimpleBinary      >("onnx::Mul"               );
        Operator::register_<SimpleBinary      >("onnx::Div"               );
        Operator::register_<SimpleBinary      >("onnx::Pow"               );
        // clang-format on
    }

}// namespace refactor::onnx
