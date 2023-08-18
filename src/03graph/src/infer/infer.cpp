#include "infer.h"
#include "common/data_type.h"
#include "common/error_handler.h"

using namespace refactor::common;

namespace refactor::graph {

    BroadcastResult multidirBroadcast(std::vector<Shape> const &inputs) {
        Shape ans;
        for (auto i = 0;; ++i) {
            auto any = false;
            len_t value = 1;
            for (auto const &input : inputs) {
                if (i < input.size()) {
                    any = true;
                    if (value == 1) {
                        value = input[i];
                    } else if (input[i] != 1 && input[i] != value) {
                        return Err(std::string("Shape broadcast failed"));
                    }
                }
            }
            if (any) {
                ans.push_back(value);
            } else {
                break;
            }
        }
        ans.shrink_to_fit();
        return Ok(std::move(ans));
    }

    bool unidirBroadcast(Shape target, Shape test) {
        if (target.size() < test.size()) {
            return false;
        } else {
            for (auto i = target.rbegin(), j = test.rbegin(); j != test.rend(); ++i, ++j) {
                if (*j != *i && *j != 1) {
                    return false;
                }
            }
            return true;
        }
    }

    InferError::InferError(std::string &&msg)
        : std::runtime_error(std::forward<std::string>(msg)) {}

    InferResult inferUnary(Edges inputs, bool typeChecker(DataType)) {
        if (inputs.size() != 1) {
            return Err(INFER_ERROR("Input size error"));
        } else if (!typeChecker(inputs[0].tensor().dataType)) {
            return Err(INFER_ERROR("Data type not support"));
        } else {
            return Ok(std::move(inputs));
        }
    }

    InferResult inferArithmetic(Edges inputs) {
        if (inputs.size() != 2) {
            return Err(INFER_ERROR("Input size error"));
        } else if (inputs[0].isTensor()) {
            auto i0 = inputs[0].tensor();
            auto i1 = inputs[1].tensor();
            if (isNumbericDataType(i0.dataType) && i0.dataType == i1.dataType) {
                auto shape = multidirBroadcast({std::move(i0.shape), std::move(i1.shape)});
                if (shape.isErr()) {
                    return Err(INFER_ERROR(shape.unwrapErr().c_str()));
                } else {
                    return Ok(Edges{EdgeInfo{Tensor{i0.dataType, shape.unwrap()}}});
                }
            } else {
                return Err(INFER_ERROR("Data type not support"));
            }
        } else {
            auto i0 = inputs[0].shapeVariable();
            auto i1 = inputs[1].shapeVariable();
            TODO("calculate shape variable");
        }
    }

    InferResult inferGemm(Edges inputs, bool transA, bool transB) {
        if (inputs.size() != 2 && inputs.size() != 3) {
            return Err(INFER_ERROR("Input size error"));
        } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
            return Err(INFER_ERROR("Edge type not support"));
        } else {
            auto a = inputs[0].tensor();
            auto b = inputs[1].tensor();
            if (a.shape.size() != 2 || b.shape.size() != 2) {
                return Err(INFER_ERROR("Input shape not support"));
            }
            auto m = transA ? a.shape[0] : a.shape[1];
            auto n = transB ? b.shape[1] : b.shape[0];
            if ((transA ? a.shape[1] : a.shape[0]) != (transB ? b.shape[0] : b.shape[1])) {
                return Err(INFER_ERROR("Input shape not support"));
            }
            if (inputs.size() == 3) {
                auto c = inputs[2].tensor();
                if (c.shape.size() != 2 || c.shape[0] != m || c.shape[1] != n) {
                    return Err(INFER_ERROR("Input shape not support"));
                }
            }
        }
    }

}// namespace refactor::graph
