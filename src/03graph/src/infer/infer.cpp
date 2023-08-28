#include "infer.h"
#include "common/data_type.h"
#include "common/error_handler.h"
#include <numeric>
#include <unordered_set>

using namespace refactor::common;

namespace refactor::graph {

    ShapeResult multidirBroadcast(std::vector<Shape> const &inputs) {
        using Iter = std::reverse_iterator<Shape::const_iterator>;
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

    ShapeResult pool(Shape const &input,
                     Shape const &kernel,
                     ShapeOrNot const &dilations,
                     ShapeOrNot const &pads,
                     ShapeOrNot const &strides) {
        auto dim = input.size();
        if (dim != kernel.size()) {
            return Err(ERROR_MSG("Input shape not support"));
        }
        Shape dilations_, pads_, strides_;
        if (dilations) {
            if (dilations->size() != dim) {
                return Err(ERROR_MSG("Input shape not support"));
            } else {
                dilations_ = *dilations;
            }
        } else {
            dilations_ = Shape(dim, 1);
        }
        if (pads) {
            if (pads->size() != dim * 2) {
                return Err(ERROR_MSG("Input shape not support"));
            } else {
                pads_ = *pads;
            }
        } else {
            pads_ = Shape(dim * 2, 0);
        }
        if (strides) {
            if (strides->size() != dim) {
                return Err(ERROR_MSG("Input shape not support"));
            } else {
                strides_ = *strides;
            }
        } else {
            strides_ = Shape(dim, 1);
        }
        Shape ans(dim);
        for (size_t i = 0; i < dim; ++i) {
            auto d = input[i] + pads_[i] + pads_[i + dim];
            auto k = (kernel[i] - 1) * dilations_[i] + 1;
            ans[i] = (d - k) / strides_[i] + 1;
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

    InferResult inferArithmetic(Edges inputs, OpType opType) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (inputs[0].isTensor() && inputs[1].isTensor()) {
            auto i0 = inputs[0].tensor();
            auto i1 = inputs[1].tensor();
            if (!isNumbericDataType(i0.dataType) || i0.dataType != i1.dataType) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            } else {
                auto shape = multidirBroadcast({std::move(i0.shape), std::move(i1.shape)});
                if (shape.isErr()) {
                    return Err(InferError(ERROR_MSG(shape.unwrapErr())));
                } else {
                    return Ok(Edges{EdgeInfo{Tensor{i0.dataType, shape.unwrap()}}});
                }
            }
        } else if (inputs[0].isShapeVariable() && inputs[1].isShapeVariable()) {
            auto i0 = inputs[0].shapeVariable().shape;
            auto i1 = inputs[1].shapeVariable().shape;
            if (i0.size() != i1.size()) {
                return Err(InferError(ERROR_MSG("Invalid shape variable")));
            } else {
                Shape ans(i0.size());
                switch (opType.underlying()) {
                    case OpType::Add:
                        for (size_t i = 0; i < ans.size(); ++i) {
                            ans[i] = i0[i] + i1[i];
                        }
                        break;
                    case OpType::Sub:
                        for (size_t i = 0; i < ans.size(); ++i) {
                            ans[i] = i0[i] - i1[i];
                        }
                        break;
                    case OpType::Mul:
                        for (size_t i = 0; i < ans.size(); ++i) {
                            ans[i] = i0[i] * i1[i];
                        }
                        break;
                    case OpType::Div:
                        for (size_t i = 0; i < ans.size(); ++i) {
                            ans[i] = i0[i] / i1[i];
                        }
                        break;
                    default:
                        return Err(InferError(ERROR_MSG("Invalid op type")));
                }
                return Ok(Edges{EdgeInfo{ShapeVariable{std::move(ans)}}});
            }
        } else {
            return Err(InferError(ERROR_MSG("Edge type not support")));
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
            auto m = transA ? a.shape[1] : a.shape[0];
            auto n = transB ? b.shape[0] : b.shape[1];
            if ((transA ? a.shape[0] : a.shape[1]) != (transB ? b.shape[1] : b.shape[0])) {
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

    InferResult inferConv(Edges inputs, ShapeOrNot dilations, ShapeOrNot pads, ShapeOrNot strides) {
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
            if (input.shape[1] % kernel.shape[1] != 0) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            Shape poolInput(dim - 2), poolKernel(dim - 2);
            std::copy(input.shape.begin() + 2, input.shape.end(), poolInput.begin());
            std::copy(kernel.shape.begin() + 2, kernel.shape.end(), poolKernel.begin());
            auto pool_ = pool(poolInput, poolKernel, dilations, pads, strides);
            if (pool_.isErr()) {
                return Err(InferError(ERROR_MSG(pool_.unwrapErr())));
            } else {
                Shape ans(dim);
                ans[0] = input.shape[0];
                ans[1] = kernel.shape[0] * (input.shape[1] / kernel.shape[1]);
                auto pool__ = pool_.unwrap();
                std::copy(pool__.begin(), pool__.end(), ans.begin() + 2);
                return Ok(Edges{EdgeInfo{Tensor{dataType, std::move(ans)}}});
            }
        }
    }

    InferResult inferPool(Edges inputs, ShapeOrNot dilations, Shape kernelShape, ShapeOrNot pads, ShapeOrNot strides) {
        if (inputs.size() != 1) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
            return Err(InferError(ERROR_MSG("Edge type not support")));
        } else {
            auto input = inputs[0].tensor();
            auto dataType = input.dataType;
            if (!isIeee754DataType(dataType)) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            auto dim = input.shape.size();
            if (dim != kernelShape.size() + 2) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            Shape inputShape(dim - 2);
            std::copy(input.shape.begin() + 2, input.shape.end(), inputShape.begin());
            auto pool_ = pool(inputShape, kernelShape, dilations, pads, strides);
            if (pool_.isErr()) {
                return Err(InferError(ERROR_MSG(pool_.unwrapErr())));
            } else {
                Shape ans(dim);
                ans[0] = input.shape[0];
                ans[1] = input.shape[1];
                auto pool__ = pool_.unwrap();
                std::copy(pool__.begin(), pool__.end(), ans.begin() + 2);
                return Ok(Edges{EdgeInfo{Tensor{dataType, std::move(ans)}}});
            }
        }
    }

    InferResult inferGlobalPool(Edges inputs) {
        if (inputs.size() != 1) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
            return Err(InferError(ERROR_MSG("Edge type not support")));
        } else {
            auto input = inputs[0].tensor();
            if (!isIeee754DataType(input.dataType)) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            auto dim = input.shape.size();
            if (dim < 2) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            Shape ans(dim, 1);
            ans[0] = input.shape[0];
            ans[1] = input.shape[1];
            return Ok(Edges{EdgeInfo{Tensor{input.dataType, std::move(ans)}}});
        }
    }

    InferResult inferReshape(Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (!inputs[0].isTensor() || !inputs[1].isShapeVariable()) {
            return Err(InferError(ERROR_MSG("Edge type not support")));
        } else {
            auto input = inputs[0].tensor();
            auto shape = inputs[1].shapeVariable().shape;
            int pos_1 = -1;
            Shape ans(shape.size());
            for (int i = 0; i < shape.size(); ++i) {
                switch (shape[i]) {
                    case -1:
                        if (pos_1 >= 0) {
                            return Err(InferError(ERROR_MSG("Invalid shape variable")));
                        }
                        pos_1 = i;
                        ans[i] = 1;
                        break;
                    case 0:
                        if (i >= input.shape.size()) {
                            return Err(InferError(ERROR_MSG("Invalid shape variable")));
                        }
                        ans[i] = input.shape[i];
                        break;
                    default:
                        ans[i] = shape[i];
                        break;
                }
            }
            auto old = std::accumulate(input.shape.begin(), input.shape.end(), 1, std::multiplies<>());
            auto now = std::accumulate(ans.begin(), ans.end(), 1, std::multiplies<>());
            if (pos_1 != -1) {
                if (old % now != 0) {
                    return Err(InferError(ERROR_MSG("Invalid shape variable")));
                } else {
                    ans[pos_1] = old / now;
                }
            } else if (old != now) {
                return Err(InferError(ERROR_MSG("Invalid shape variable")));
            }
            return Ok(Edges{EdgeInfo{Tensor{input.dataType, std::move(ans)}}});
        }
    }

    InferResult inferBatchNormalization(Edges inputs, bool training) {
        if (inputs.size() != 5) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
            return Err(InferError(ERROR_MSG("Edge type not support")));
        } else {
            auto input = inputs[0].tensor();
            auto scale = inputs[1].tensor();
            auto bias = inputs[2].tensor();
            auto mean = inputs[3].tensor();
            auto variance = inputs[4].tensor();
            DataType dt[]{input.dataType, scale.dataType, mean.dataType};
            if (!std::all_of(dt, dt + 3, isFloatDataType) || bias.dataType != dt[1] || variance.dataType != dt[2]) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (input.shape.size() < 1 || scale.shape != bias.shape || scale.shape != mean.shape || scale.shape != variance.shape) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            if (!training) {
                return Ok(Edges{std::move(input)});
            } else {
                return Err(InferError(ERROR_MSG("Not implemented")));
            }
        }
    }

    InferResult inferTranspose(Edges inputs, ShapeOrNot perms) {
        if (inputs.size() != 1) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (!inputs[0].isTensor()) {
            return Err(InferError(ERROR_MSG("Invalid input edge type")));
        } else if (!isIeee754DataType(inputs[0].tensor().dataType)) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        } else {
            auto input = inputs[0].tensor();
            int rank = input.shape.size();
            Shape ans(rank);
            Shape perm;
            //TODO: initialize is right or wrong here
            if (perms.has_value()) {
                perm = perms.value();
            } else {
                for (int i = 0; i < rank; ++i) {
                    perm[i] = i;
                }
            }
            if (rank != perm.size()) {
                return Err(InferError(ERROR_MSG("perm size not equal to rank")));
            }
            for (int i = 0; i < rank; ++i) {
                if (perm[i] >= rank) {
                    return Err(InferError(ERROR_MSG("perm bigger than rank")));
                }
                ans[i] = input.shape[perm[i]];
            }
            return Ok(Edges{EdgeInfo{Tensor{input.dataType, std::move(ans)}}});
        }
    }

}// namespace refactor::graph
