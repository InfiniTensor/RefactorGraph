#include "infer.h"
#include "common/data_type.h"
#include "common/error_handler.h"
#include <unordered_set>

using namespace refactor::common;

namespace refactor::graph {

    BroadcastResult multidirBroadcast(std::vector<Shape> const &inputs) {
        using Iter = std::reverse_iterator<std::vector<len_t>::const_iterator>;
        std::vector<std::pair<Iter, Iter>> iters;
        iters.reserve(inputs.size());
        for (auto const &input : inputs) {
            iters.emplace_back(input.rbegin(), input.rend());
        }
        Shape ans;
        while (true) {
            std::unordered_set<len_t> dims;
            for (auto &[begin, end] : iters) {
                if (begin != end) {
                    dims.insert(*begin++);
                }
            }
            if (dims.size() == 0) {
                break;
            } else if (dims.size() == 1 || (dims.size() == 2 && dims.erase(1) == 1)) {
                ans.push_back(*dims.begin());
            } else {
                return Err(BROADCAST_ERROR("Shape broadcast failed"));
            }
        }
        std::reverse(ans.begin(), ans.end());
        return Ok(ans);
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
            auto dataType = a.dataType;
            if (!isNumbericDataType(dataType) || b.dataType != dataType) {
                return Err(INFER_ERROR("Input data type not support"));
            }
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
                if (c.dataType != dataType) {
                    return Err(INFER_ERROR("Input data type not support"));
                }
                if (c.shape.size() != 2 || !unidirBroadcast({m, n}, c.shape)) {
                    return Err(INFER_ERROR("Input shape not support"));
                }
            }
            return Ok(Edges{EdgeInfo{Tensor{dataType, {m, n}}}});
        }
    }

}// namespace refactor::graph
