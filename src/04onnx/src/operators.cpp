#include "onnx/operators.h"
#include "frontend/operator.h"
#include "operators/common.h"

namespace refactor::onnx {

    void register_() {
        // clang-format off
        OpType::register_("onnx::Constant"        , inferConstant        , unreachableLower );
        OpType::register_("onnx::ConstantOfShape" , inferConstantOfShape , unreachableLower );
        OpType::register_("onnx::Relu"            , inferUnary           , lowerUnary       );
        OpType::register_("onnx::Sqrt"            , inferUnary           , lowerUnary       );
        OpType::register_("onnx::Tanh"            , inferUnary           , lowerUnary       );
        OpType::register_("onnx::Add"             , inferArithmetic      , lowerArithmetic  );
        OpType::register_("onnx::Sub"             , inferArithmetic      , lowerArithmetic  );
        OpType::register_("onnx::Mul"             , inferArithmetic      , lowerArithmetic  );
        OpType::register_("onnx::Div"             , inferArithmetic      , lowerArithmetic  );
        OpType::register_("onnx::Pow"             , inferPow             , lowerPow         );
        OpType::register_("onnx::MatMul"          , inferMatMul          , lowerMatMul      );
        OpType::register_("onnx::Gemm"            , inferGemm            , lowerGemm        );
        OpType::register_("onnx::CumSum"          , inferCumSum          , lowerCumSum      );
        OpType::register_("onnx::Max"             , inferSelect          , lowerSelect      );
        OpType::register_("onnx::Min"             , inferSelect          , lowerSelect      );
        OpType::register_("onnx::Transpose"       , inferTranspose       , lowerTranspose   );
        OpType::register_("onnx::Cast"            , inferCast            , lowerCast        );
        OpType::register_("onnx::Range"           , inferRange           , lowerRange       );
        OpType::register_("onnx::Slice"           , inferSlice           , lowerSlice       );
        OpType::register_("onnx::Split"           , inferSplit           , lowerSplit       );
        OpType::register_("onnx::Shape"           , inferShape           , unreachableLower );
        OpType::register_("onnx::Reshape"         , inferReshape         , lowerReshape     );
        OpType::register_("onnx::Gather"          , inferGather          , lowerGather      );
        OpType::register_("onnx::Concat"          , inferConcat          , lowerConcat      );
        OpType::register_("onnx::Expand"          , inferExpand          , lowerExpand      );
        OpType::register_("onnx::Where"           , inferWhere           , lowerWhere       );
        OpType::register_("onnx::Squeeze"         , inferSqueeze         , lowerSqueeze     );
        OpType::register_("onnx::Unsqueeze"       , inferUnsqueeze       , lowerUnsqueeze   );
        OpType::register_("onnx::Equal"           , inferCompair         , lowerCompair     );
        OpType::register_("onnx::Greater"         , inferCompair         , lowerCompair     );
        OpType::register_("onnx::GreaterOrEqual"  , inferCompair         , lowerCompair     );
        OpType::register_("onnx::Less"            , inferCompair         , lowerCompair     );
        OpType::register_("onnx::LessOrEqual"     , inferCompair         , lowerCompair     );
        OpType::register_("onnx::Softmax"         , inferSoftmax         , lowerSoftmax     );
        OpType::register_("onnx::ReduceMean"      , inferReduce          , lowerReduce      );
        // clang-format on
    }

}// namespace refactor::onnx
