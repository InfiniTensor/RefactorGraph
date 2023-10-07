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
#include "operators/range.hh"
#include "operators/reduce.hh"
#include "operators/reshape.hh"
#include "operators/select.hh"
#include "operators/shape.hh"
#include "operators/simple_binary.hh"
#include "operators/slice.hh"
#include "operators/softmax.hh"
#include "operators/split.hh"

namespace refactor::onnx {

    void register_() {
        // clang-format off
        #define REGISTER(NAME, CLASS) Operator::register_<CLASS>("onnx::" #NAME)
        REGISTER(BatchNormalization, BatchNormalization);
        REGISTER(Cast              , Cast              );
        REGISTER(Equal             , Compair           );
        REGISTER(Greater           , Compair           );
        REGISTER(GreaterOrEqual    , Compair           );
        REGISTER(Less              , Compair           );
        REGISTER(LessOrEqual       , Compair           );
        REGISTER(Concat            , Concat            );
        REGISTER(Constant          , Constant          );
        REGISTER(ConstantOfShape   , ConstantOfShape   );
        REGISTER(Conv              , Conv              );
        REGISTER(CumSum            , CumSum            );
        REGISTER(Expand            , Expand            );
        REGISTER(Gather            , Gather            );
        REGISTER(GatherElements    , GatherElements    );
        REGISTER(Gemm              , Gemm              );
        REGISTER(GlobalAveragePool , GlobalPool        );
        REGISTER(GlobalLpPool      , GlobalPool        );
        REGISTER(GlobalMaxPool     , GlobalPool        );
        REGISTER(MatMul            , MatMul            );
        REGISTER(AveragePool       , Pool              );
        REGISTER(LpPool            , Pool              );
        REGISTER(MaxPool           , Pool              );
        REGISTER(Range             , Range             );
        REGISTER(ReduceMean        , Reduce            );
        REGISTER(ReduceL1          , Reduce            );
        REGISTER(ReduceL2          , Reduce            );
        REGISTER(ReduceLogSum      , Reduce            );
        REGISTER(ReduceLogSumExp   , Reduce            );
        REGISTER(ReduceMax         , Reduce            );
        REGISTER(ReduceMin         , Reduce            );
        REGISTER(ReduceProd        , Reduce            );
        REGISTER(ReduceSum         , Reduce            );
        REGISTER(ReduceSumSquare   , Reduce            );
        REGISTER(Reshape           , Reshape           );
        REGISTER(Max               , Select            );
        REGISTER(Min               , Select            );
        REGISTER(Shape             , Shape             );
        REGISTER(Add               , SimpleBinary      );
        REGISTER(Sub               , SimpleBinary      );
        REGISTER(Mul               , SimpleBinary      );
        REGISTER(Div               , SimpleBinary      );
        REGISTER(Pow               , SimpleBinary      );
        REGISTER(Slice             , Slice             );
        REGISTER(Softmax           , Softmax           );
        REGISTER(Split             , Split             );
        #undef REGISTER
        // clang-format on
    }

}// namespace refactor::onnx
