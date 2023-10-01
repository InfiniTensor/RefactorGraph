#include "onnx/operators.h"
#include "frontend/operator.h"
#include "operators/common.h"

namespace refactor::onnx {

    void register_() {
        // clang-format off
        OpType::register_("onnx::Add"                , inferArithmetic         , lowerArithmetic         );
        OpType::register_("onnx::Sub"                , inferArithmetic         , lowerArithmetic         );
        OpType::register_("onnx::Mul"                , inferArithmetic         , lowerArithmetic         );
        OpType::register_("onnx::Div"                , inferArithmetic         , lowerArithmetic         );
        OpType::register_("onnx::BatchNormalization" , inferBatchNormalization , lowerBatchNormalization );
        OpType::register_("onnx::Cast"               , inferCast               , lowerCast               );
        OpType::register_("onnx::Equal"              , inferCompair            , lowerCompair            );
        OpType::register_("onnx::Greater"            , inferCompair            , lowerCompair            );
        OpType::register_("onnx::GreaterOrEqual"     , inferCompair            , lowerCompair            );
        OpType::register_("onnx::Less"               , inferCompair            , lowerCompair            );
        OpType::register_("onnx::LessOrEqual"        , inferCompair            , lowerCompair            );
        OpType::register_("onnx::Concat"             , inferConcat             , lowerConcat             );
        OpType::register_("onnx::ConstantOfShape"    , inferConstantOfShape    , unreachableLower        );
        OpType::register_("onnx::Constant"           , inferConstant           , unreachableLower        );
        OpType::register_("onnx::Conv"               , inferConv               , lowerConv               );
        OpType::register_("onnx::CumSum"             , inferCumSum             , lowerCumSum             );
        OpType::register_("onnx::Expand"             , inferExpand             , lowerExpand             );
        OpType::register_("onnx::GatherElements"     , inferGatherElements     , lowerGatherElements     );
        OpType::register_("onnx::Gather"             , inferGather             , lowerGather             );
        OpType::register_("onnx::Gemm"               , inferGemm               , lowerGemm               );
        OpType::register_("onnx::GlobalAveragePool"  , inferGlobalPool         , lowerGlobalPool         );
        OpType::register_("onnx::GlobalLpPool"       , inferGlobalPool         , lowerGlobalPool         );
        OpType::register_("onnx::GlobalMaxPool"      , inferGlobalPool         , lowerGlobalPool         );
        OpType::register_("onnx::Not"                , inferLogic              , lowerLogic              );
        OpType::register_("onnx::And"                , inferLogic              , lowerLogic              );
        OpType::register_("onnx::Or"                 , inferLogic              , lowerLogic              );
        OpType::register_("onnx::Xor"                , inferLogic              , lowerLogic              );
        OpType::register_("onnx::MatMul"             , inferMatMul             , lowerMatMul             );
        OpType::register_("onnx::AveragePool"        , inferPool               , lowerPool               );
        OpType::register_("onnx::LpPool"             , inferPool               , lowerPool               );
        OpType::register_("onnx::MaxPool"            , inferPool               , lowerPool               );
        OpType::register_("onnx::Pow"                , inferPow                , lowerPow                );
        OpType::register_("onnx::Range"              , inferRange              , unreachableLower        );
        OpType::register_("onnx::ReduceMean"         , inferReduce             , lowerReduce             );
        OpType::register_("onnx::Reshape"            , inferReshape            , lowerReshape            );
        OpType::register_("onnx::Max"                , inferSelect             , lowerSelect             );
        OpType::register_("onnx::Min"                , inferSelect             , lowerSelect             );
        OpType::register_("onnx::Shape"              , inferShape              , unreachableLower        );
        OpType::register_("onnx::Slice"              , inferSlice              , lowerSlice              );
        OpType::register_("onnx::Softmax"            , inferSoftmax            , lowerSoftmax            );
        OpType::register_("onnx::Split"              , inferSplit              , lowerSplit              );
        OpType::register_("onnx::Squeeze"            , inferSqueeze            , lowerSqueeze            );
        OpType::register_("onnx::Tile"               , inferTile               , lowerTile               );
        OpType::register_("onnx::Transpose"          , inferTranspose          , lowerTranspose          );
        OpType::register_("onnx::Erf"                , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Relu"               , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Sigmoid"            , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Sqrt"               , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Tanh"               , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Identity"           , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Cos"                , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Sin"                , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Abs"                , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Neg"                , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Log"                , inferUnary              , lowerUnary              );
        OpType::register_("onnx::Unsqueeze"          , inferUnsqueeze          , lowerUnsqueeze          );
        OpType::register_("onnx::Where"              , inferWhere              , lowerWhere              );
        // clang-format on
    }

}// namespace refactor::onnx
