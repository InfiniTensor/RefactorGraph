#include "onnx/operators.h"
#include "operators/batch_normalization.hh"
#include "operators/cast.hh"
#include "operators/clip.hh"
#include "operators/compair.hh"
#include "operators/concat.hh"
#include "operators/constant.hh"
#include "operators/constant_of_shape.hh"
#include "operators/conv.hh"
#include "operators/cum_sum.hh"
#include "operators/depth_to_space.hh"
#include "operators/dequantize_linear.hh"
#include "operators/dynamic_quantize_linear.hh"
#include "operators/einsum.hh"
#include "operators/expand.hh"
#include "operators/flatten.hh"
#include "operators/gather.hh"
#include "operators/gather_elements.hh"
#include "operators/gemm.hh"
#include "operators/global_pool.hh"
#include "operators/hard_sigmoid.hh"
#include "operators/layernorm.hh"
#include "operators/mat_mul.hh"
#include "operators/mat_mul_integer.hh"
#include "operators/pad.hh"
#include "operators/pool.hh"
#include "operators/range.hh"
#include "operators/reduce.hh"
#include "operators/reshape.hh"
#include "operators/scatter_nd.hh"
#include "operators/select.hh"
#include "operators/shape.hh"
#include "operators/simple_binary.hh"
#include "operators/simple_unary.hh"
#include "operators/slice.hh"
#include "operators/softmax.hh"
#include "operators/split.hh"
#include "operators/squeeze.hh"
#include "operators/tile.hh"
#include "operators/transpose.hh"
#include "operators/unsqueeze.hh"
#include "operators/where.hh"

namespace refactor::onnx {

    void register_() {
#define REGISTER(NAME, CLASS) Operator::register_<CLASS>("onnx::" #NAME)
        // clang-format off
        REGISTER(BatchNormalization   , BatchNormalization   );
        REGISTER(Cast                 , Cast                 );
        REGISTER(Clip                 , Clip                 );
        REGISTER(Equal                , Compair              );
        REGISTER(Greater              , Compair              );
        REGISTER(GreaterOrEqual       , Compair              );
        REGISTER(Less                 , Compair              );
        REGISTER(LessOrEqual          , Compair              );
        REGISTER(Concat               , Concat               );
        REGISTER(Constant             , Constant             );
        REGISTER(ConstantOfShape      , ConstantOfShape      );
        REGISTER(Conv                 , Conv                 );
        REGISTER(DequantizeLinear     , DequantizeLinear     );
        REGISTER(DynamicQuantizeLinear, DynamicQuantizeLinear);
        REGISTER(CumSum               , CumSum               );
        REGISTER(Einsum               , Einsum               );
        REGISTER(Expand               , Expand               );
        REGISTER(Gather               , Gather               );
        REGISTER(GatherElements       , GatherElements       );
        REGISTER(Gemm                 , Gemm                 );
        REGISTER(GlobalAveragePool    , GlobalPool           );
        REGISTER(GlobalLpPool         , GlobalPool           );
        REGISTER(GlobalMaxPool        , GlobalPool           );
        REGISTER(MatMul               , MatMul               );
        REGISTER(MatMulInteger        , MatMulInteger        );
        REGISTER(AveragePool          , Pool                 );
        REGISTER(LpPool               , Pool                 );
        REGISTER(MaxPool              , Pool                 );
        REGISTER(Range                , Range                );
        REGISTER(ReduceMean           , Reduce               );
        REGISTER(ReduceL1             , Reduce               );
        REGISTER(ReduceL2             , Reduce               );
        REGISTER(ReduceLogSum         , Reduce               );
        REGISTER(ReduceLogSumExp      , Reduce               );
        REGISTER(ReduceMax            , Reduce               );
        REGISTER(ReduceMin            , Reduce               );
        REGISTER(ReduceProd           , Reduce               );
        REGISTER(ReduceSum            , Reduce               );
        REGISTER(ReduceSumSquare      , Reduce               );
        REGISTER(Reshape              , Reshape              );
        REGISTER(Flatten              , Flatten              );
        REGISTER(ScatterND            , ScatterND            );
        REGISTER(Max                  , Select               );
        REGISTER(Min                  , Select               );
        REGISTER(Shape                , Shape                );
        REGISTER(Add                  , SimpleBinary         );
        REGISTER(Sub                  , SimpleBinary         );
        REGISTER(Mul                  , SimpleBinary         );
        REGISTER(Div                  , SimpleBinary         );
        REGISTER(Pow                  , SimpleBinary         );
        REGISTER(And                  , SimpleBinary         );
        REGISTER(Or                   , SimpleBinary         );
        REGISTER(Xor                  , SimpleBinary         );
        REGISTER(Mod                  , SimpleBinary         );
        REGISTER(Abs                  , SimpleUnary          );
        REGISTER(Acos                 , SimpleUnary          );
        REGISTER(Acosh                , SimpleUnary          );
        REGISTER(Asin                 , SimpleUnary          );
        REGISTER(Asinh                , SimpleUnary          );
        REGISTER(Atan                 , SimpleUnary          );
        REGISTER(Atanh                , SimpleUnary          );
        REGISTER(Cos                  , SimpleUnary          );
        REGISTER(Cosh                 , SimpleUnary          );
        REGISTER(Sin                  , SimpleUnary          );
        REGISTER(Sinh                 , SimpleUnary          );
        REGISTER(Tan                  , SimpleUnary          );
        REGISTER(Tanh                 , SimpleUnary          );
        REGISTER(Relu                 , SimpleUnary          );
        REGISTER(Sqrt                 , SimpleUnary          );
        REGISTER(Sigmoid              , SimpleUnary          );
        REGISTER(Erf                  , SimpleUnary          );
        REGISTER(Log                  , SimpleUnary          );
        REGISTER(Not                  , SimpleUnary          );
        REGISTER(Neg                  , SimpleUnary          );
        REGISTER(Identity             , SimpleUnary          );
        REGISTER(HardSwish            , SimpleUnary          );
        REGISTER(Exp                  , SimpleUnary          );
        REGISTER(Slice                , Slice                );
        REGISTER(Softmax              , Softmax              );
        REGISTER(Split                , Split                );
        REGISTER(Squeeze              , Squeeze              );
        REGISTER(Tile                 , Tile                 );
        REGISTER(Transpose            , Transpose            );
        REGISTER(Unsqueeze            , Unsqueeze            );
        REGISTER(Where                , Where                );
        REGISTER(HardSigmoid          , HardSigmoid          );
        REGISTER(Pad                  , Pad                  );
        REGISTER(DepthToSpace         , DepthToSpace         );
        REGISTER(LayerNormalization   , Layernorm           );
        // clang-format on
#undef REGISTER
    }

}// namespace refactor::onnx
