#include "infer.h"
#include "common/data_type.h"
#include "common/error_handler.h"
#include <unordered_set>

using namespace refactor::common;

namespace refactor::graph {

    ShapeResult multidirBroadcast(std::vector<Shape> const &inputs) {
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
                return Err(ERROR_MSG("Shape broadcast failed"));
            }
        }
        std::reverse(ans.begin(), ans.end());
        return Ok(ans);
    }

    bool unidirBroadcast(Shape const &target, Shape const &test) {
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

    ShapeResult pool(Shape const &input, Shape const &kernel, Shape const &dilations, Shape const &pads, Shape const &strides) {
        auto dim = input.size();
        if (dim != kernel.size() || dim != dilations.size() || dim != pads.size() / 2 || dim != strides.size()) {
            return Err(ERROR_MSG("Input shape not support"));
        }
        Shape ans(dim);
        for (size_t i = 0; i < dim; ++i) {
            auto d = input[i] + pads[i] + pads[i + dim];
            auto k = (kernel[i] - 1) * dilations[i] + 1;
            ans[i] = (d - k) / strides[i] + 1;
        }
        return Ok(ans);
    }

    InferError::InferError(std::string &&msg)
        : std::runtime_error(std::forward<std::string>(msg)) {}

    InferResult inferUnary(Edges inputs, bool typeChecker(DataType)) {
        if (inputs.size() != 1) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (!typeChecker(inputs[0].tensor().dataType)) {
            return Err(InferError(ERROR_MSG("Data type not support")));
        } else {
            return Ok(std::move(inputs));
        }
    }

    InferResult inferArithmetic(Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (inputs[0].isTensor()) {
            auto i0 = inputs[0].tensor();
            auto i1 = inputs[1].tensor();
            if (isNumbericDataType(i0.dataType) && i0.dataType == i1.dataType) {
                auto shape = multidirBroadcast({std::move(i0.shape), std::move(i1.shape)});
                if (shape.isErr()) {
                    return Err(InferError(ERROR_MSG(shape.unwrapErr())));
                } else {
                    return Ok(Edges{EdgeInfo{Tensor{i0.dataType, shape.unwrap()}}});
                }
            } else {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
        } else {
            auto i0 = inputs[0].shapeVariable();
            auto i1 = inputs[1].shapeVariable();
            TODO("calculate shape variable");
        }
    }

    InferResult inferGemm(Edges inputs, bool transA, bool transB) {
        if (inputs.size() != 2 && inputs.size() != 3) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
            return Err(InferError(ERROR_MSG("Edge type not support")));
        } else {
            auto a = inputs[0].tensor();
            auto b = inputs[1].tensor();
            auto dataType = a.dataType;
            if (!isNumbericDataType(dataType) || b.dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (a.shape.size() != 2 || b.shape.size() != 2) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            auto m = transA ? a.shape[0] : a.shape[1];
            auto n = transB ? b.shape[1] : b.shape[0];
            if ((transA ? a.shape[1] : a.shape[0]) != (transB ? b.shape[0] : b.shape[1])) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            if (inputs.size() == 3) {
                auto c = inputs[2].tensor();
                if (c.dataType != dataType) {
                    return Err(InferError(ERROR_MSG("Input data type not support")));
                }
                if (c.shape.size() != 2 || !unidirBroadcast({m, n}, c.shape)) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
            }
            return Ok(Edges{EdgeInfo{Tensor{dataType, {m, n}}}});
        }
    }

    InferResult inferConv(Edges inputs, Shape dilations, len_t group, Shape pads, Shape strides) {
        if (inputs.size() != 2 && inputs.size() != 3) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
            return Err(InferError(ERROR_MSG("Edge type not support")));
        } else {
            auto input = inputs[0].tensor();
            auto kernel = inputs[1].tensor();
            auto dataType = input.dataType;
            if (!isIeee754DataType(dataType) || kernel.dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (inputs.size() > 2) {
                if (!inputs[2].isTensor()) {
                    return Err(InferError(ERROR_MSG("Edge type not support")));
                }
                auto bias = inputs[2].tensor();
                if (bias.dataType != dataType) {
                    return Err(InferError(ERROR_MSG("Input data type not support")));
                }
                if (bias.shape.size() != 1 || bias.shape[0] != kernel.shape[0]) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
            }
            auto dim = input.shape.size();
            if (dim < 2 || dim != kernel.shape.size()) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            Shape ans(dim);
            ans[0] = input.shape[0];
            ans[1] = kernel.shape[0] * group;
            Shape poolInput(dim - 2), poolKernel(dim - 2);
            std::copy(input.shape.begin() + 2, input.shape.end(), poolInput.begin());
            std::copy(kernel.shape.begin() + 2, kernel.shape.end(), poolKernel.begin());
            auto pool_ = pool(poolInput, poolKernel, dilations, pads, strides);
            if (pool_.isErr()) {
                return Err(InferError(ERROR_MSG(pool_.unwrapErr())));
            } else {
                auto pool__ = pool_.unwrap();
                std::copy(pool__.begin(), pool__.end(), ans.begin() + 2);
            }
            return Ok(Edges{EdgeInfo{Tensor{dataType, std::move(ans)}}});
        }
    }

    InferResult inferPool(Edges inputs, Shape dilations, Shape kernelShape, Shape pads, Shape strides) {
        if (inputs.size() != 1) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
            return Err(InferError(ERROR_MSG("Edge type not support")));
        } else {
            auto input = inputs[0].tensor();
            auto kernel = inputs[1].tensor();
            auto dataType = input.dataType;
            if (!isIeee754DataType(dataType) || kernel.dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            auto dim = input.shape.size();
            if (dim != kernel.shape.size()) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            Shape ans(dim);
            auto pool_ = pool(input.shape, kernelShape, dilations, pads, strides);
            if (pool_.isErr()) {
                return Err(InferError(ERROR_MSG(pool_.unwrapErr())));
            } else {
                return Ok(Edges{EdgeInfo{Tensor{dataType, std::move(pool_.unwrap())}}});
            }
        }
    }

    InferResult inferGlobalPool(Edges) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferReshape(Edges) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferBatchNormalization(Edges) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

}// namespace refactor::graph
