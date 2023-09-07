#include "infer.h"
#include "common/op_type.h"
#include <vector>

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
            std::unordered_set<size_t> dims;
            for (auto &[begin, end] : iters) {
                if (begin != end) {
                    dims.insert(begin++->value());
                }
            }
            if (dims.size() == 0) {
                break;
            } else if (dims.size() == 1 || (dims.size() == 2 && dims.erase(1) == 1)) {
                ans.emplace_back(*dims.begin());
            } else {
                return Err(ERROR_MSG("Shape broadcast failed"));
            }
        }
        std ::reverse(ans.begin(), ans.end());
        return Ok(ans);
    }

    bool unidirBroadcast(Shape const &target, Shape const &test) {
        if (target.size() < test.size()) {
            return false;
        } else {
            for (auto i = target.rbegin(), j = test.rbegin(); j != test.rend(); ++i, ++j) {
                if (*j != *i && *j != DimExpr(1)) {
                    return false;
                }
            }
            return true;
        }
    }

    // ShapeResult pool(Shape const &input,
    //                  Shape const &kernel,
    //                  ShapeOrNot const &dilations,
    //                  ShapeOrNot const &pads,
    //                  ShapeOrNot const &strides) {
    //     auto dim = input.size();
    //     if (dim != kernel.size()) {
    //         return Err(ERROR_MSG("Input shape not support"));
    //     }
    //     Shape dilations_, pads_, strides_;
    //     if (dilations) {
    //         if (dilations->size() != dim) {
    //             return Err(ERROR_MSG("Input shape not support"));
    //         } else {
    //             dilations_ = *dilations;
    //         }
    //     } else {
    //         dilations_ = Shape(dim, 1);
    //     }
    //     if (pads) {
    //         if (pads->size() != dim * 2) {
    //             return Err(ERROR_MSG("Input shape not support"));
    //         } else {
    //             pads_ = *pads;
    //         }
    //     } else {
    //         pads_ = Shape(dim * 2, 0);
    //     }
    //     if (strides) {
    //         if (strides->size() != dim) {
    //             return Err(ERROR_MSG("Input shape not support"));
    //         } else {
    //             strides_ = *strides;
    //         }
    //     } else {
    //         strides_ = Shape(dim, 1);
    //     }
    //     Shape ans(dim);
    //     for (size_t i = 0; i < dim; ++i) {
    //         auto d = input[i] + pads_[i] + pads_[i + dim];
    //         auto k = (kernel[i] - 1) * dilations_[i] + 1;
    //         ans[i] = (d - k) / strides_[i] + 1;
    //     }
    //     return Ok(ans);
    // }

    InferError::InferError(std::string &&msg)
        : std::runtime_error(std::forward<std::string>(msg)) {}

    InferResult inferUnary(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 1) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (inputs[0]->hasData()) {
            TODO("Not implemented");
        } else {
            auto dataType = inputs[0]->dataType;
            switch (node.operator_().opType.underlying()) {
                case OpType::Abs:
                case OpType::Relu:
                case OpType::PRelu:
                    if (!isNumbericDataType(dataType)) {
                        return Err(InferError(ERROR_MSG("Data type not support")));
                    }
                    break;
                case OpType::Acos:
                case OpType::Acosh:
                case OpType::Asin:
                case OpType::Asinh:
                case OpType::Atan:
                case OpType::Atanh:
                case OpType::Cos:
                case OpType::Cosh:
                case OpType::Sin:
                case OpType::Sinh:
                case OpType::Tan:
                    if (!isIeee754DataType(dataType)) {
                        return Err(InferError(ERROR_MSG("Data type not support")));
                    }
                    break;
                case OpType::Tanh:
                    if (!isFloatDataType(dataType)) {
                        return Err(InferError(ERROR_MSG("Data type not support")));
                    }
                    break;
                default:
                    TODO("Not implemented");
            }
            return Ok(std::move(inputs));
        }
    }

    InferResult inferArithmetic(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (inputs[0]->hasData() && inputs[1]->hasData()) {
            TODO("Not implemented");
        } else {
            auto dataType = inputs[0]->dataType;
            if (!isNumbericDataType(dataType) || inputs[1]->dataType != dataType) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            } else {
                auto shape = multidirBroadcast({inputs[0]->shape, inputs[1]->shape});
                if (shape.isErr()) {
                    return Err(InferError(ERROR_MSG(shape.unwrapErr())));
                } else {
                    return Ok(Edges{std::make_shared<Tensor>(dataType, shape.unwrap())});
                }
            }
        }
    }

    InferResult inferGemm(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2 && inputs.size() != 3) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (std::all_of(inputs.begin(), inputs.end(), [](auto const &edge) { return edge->hasData(); })) {
            TODO("Not implemented");
        } else {
            auto a = inputs[0];
            auto b = inputs[1];
            auto dataType = a->dataType;
            if (!isNumbericDataType(dataType) || b->dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (a->shape.size() != 2 || b->shape.size() != 2) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }

            auto const &attrs = node.operator_().attributes;
            size_t m, n, k;
            if (auto it = attrs.find("transA"); it == attrs.end() || it->second.int_() == 0) {
                m = a->shape[0].value();
                k = a->shape[1].value();
            } else {
                m = a->shape[1].value();
                k = a->shape[0].value();
            }
            if (auto it = attrs.find("transB"); it == attrs.end() || it->second.int_() == 0) {
                if (b->shape[0].value() != k) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                n = b->shape[1].value();
            } else {
                if (b->shape[1].value() != k) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                n = b->shape[0].value();
            }
            Shape ans;
            if (inputs.size() == 3) {
                auto c = inputs[2];
                if (c->dataType != dataType) {
                    return Err(InferError(ERROR_MSG("Input data type not support")));
                }
                if (c->shape.size() != 2 || !unidirBroadcast(Shape{DimExpr(m), DimExpr(n)}, c->shape)) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
            }
            return Ok(Edges{std::make_shared<Tensor>(dataType, Shape{DimExpr(m), DimExpr(n)})});
        }
    }

    InferResult inferMatMul(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferReshape(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferCumSum(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferSlice(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferShape(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferWhere(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferSqueeze(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferEqual(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferSoftmax(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferPow(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferReduce(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferConcat(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferGather(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferCast(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferUnsqueeze(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferMax(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferTranspose(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferConstantOfShape(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

    InferResult inferExpand(NodeInfo const &node, Edges inputs) {
        return Err(InferError(ERROR_MSG("Not implemented")));
    }

}// namespace refactor::graph


//     InferResult inferConv(Edges inputs, ShapeOrNot dilations, ShapeOrNot pads, ShapeOrNot strides) {
//         if (inputs.size() != 2 && inputs.size() != 3) {
//             return Err(InferError(ERROR_MSG("Input size error")));
//         } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
//             return Err(InferError(ERROR_MSG("Edge type not support")));
//         } else {
//             auto input = inputs[0].tensor();
//             auto kernel = inputs[1].tensor();
//             auto dataType = input.dataType;
//             if (!isIeee754DataType(dataType) || kernel.dataType != dataType) {
//                 return Err(InferError(ERROR_MSG("Input data type not support")));
//             }
//             if (inputs.size() > 2) {
//                 if (!inputs[2].isTensor()) {
//                     return Err(InferError(ERROR_MSG("Edge type not support")));
//                 }
//                 auto bias = inputs[2].tensor();
//                 if (bias.dataType != dataType) {
//                     return Err(InferError(ERROR_MSG("Input data type not support")));
//                 }
//                 if (bias.shape.size() != 1 || bias.shape[0] != kernel.shape[0]) {
//                     return Err(InferError(ERROR_MSG("Input shape not support")));
//                 }
//             }
//             auto dim = input.shape.size();
//             if (dim < 2 || dim != kernel.shape.size()) {
//                 return Err(InferError(ERROR_MSG("Input shape not support")));
//             }
//             if (input.shape[1] % kernel.shape[1] != 0) {
//                 return Err(InferError(ERROR_MSG("Input shape not support")));
//             }
//             Shape poolInput(dim - 2), poolKernel(dim - 2);
//             std::copy(input.shape.begin() + 2, input.shape.end(), poolInput.begin());
//             std::copy(kernel.shape.begin() + 2, kernel.shape.end(), poolKernel.begin());
//             auto pool_ = pool(poolInput, poolKernel, dilations, pads, strides);
//             if (pool_.isErr()) {
//                 return Err(InferError(ERROR_MSG(pool_.unwrapErr())));
//             } else {
//                 Shape ans(dim);
//                 ans[0] = input.shape[0];
//                 ans[1] = kernel.shape[0] * (input.shape[1] / kernel.shape[1]);
//                 auto pool__ = pool_.unwrap();
//                 std::copy(pool__.begin(), pool__.end(), ans.begin() + 2);
//                 return Ok(Edges{EdgeInfo{Tensor{dataType, std::move(ans)}}});
//             }
//         }
//     }

//     InferResult inferPool(Edges inputs, ShapeOrNot dilations, Shape kernelShape, ShapeOrNot pads, ShapeOrNot strides) {
//         if (inputs.size() != 1) {
//             return Err(InferError(ERROR_MSG("Input size error")));
//         } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
//             return Err(InferError(ERROR_MSG("Edge type not support")));
//         } else {
//             auto input = inputs[0].tensor();
//             auto dataType = input.dataType;
//             if (!isIeee754DataType(dataType)) {
//                 return Err(InferError(ERROR_MSG("Input data type not support")));
//             }
//             auto dim = input.shape.size();
//             if (dim != kernelShape.size() + 2) {
//                 return Err(InferError(ERROR_MSG("Input shape not support")));
//             }
//             Shape inputShape(dim - 2);
//             std::copy(input.shape.begin() + 2, input.shape.end(), inputShape.begin());
//             auto pool_ = pool(inputShape, kernelShape, dilations, pads, strides);
//             if (pool_.isErr()) {
//                 return Err(InferError(ERROR_MSG(pool_.unwrapErr())));
//             } else {
//                 Shape ans(dim);
//                 ans[0] = input.shape[0];
//                 ans[1] = input.shape[1];
//                 auto pool__ = pool_.unwrap();
//                 std::copy(pool__.begin(), pool__.end(), ans.begin() + 2);
//                 return Ok(Edges{EdgeInfo{Tensor{dataType, std::move(ans)}}});
//             }
//         }
//     }

//     InferResult inferGlobalPool(Edges inputs) {
//         if (inputs.size() != 1) {
//             return Err(InferError(ERROR_MSG("Input size error")));
//         } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
//             return Err(InferError(ERROR_MSG("Edge type not support")));
//         } else {
//             auto input = inputs[0].tensor();
//             if (!isIeee754DataType(input.dataType)) {
//                 return Err(InferError(ERROR_MSG("Input data type not support")));
//             }
//             auto dim = input.shape.size();
//             if (dim < 2) {
//                 return Err(InferError(ERROR_MSG("Input shape not support")));
//             }
//             Shape ans(dim, 1);
//             ans[0] = input.shape[0];
//             ans[1] = input.shape[1];
//             return Ok(Edges{EdgeInfo{Tensor{input.dataType, std::move(ans)}}});
//         }
//     }

//     InferResult inferReshape(Edges inputs) {
//         if (inputs.size() != 2) {
//             return Err(InferError(ERROR_MSG("Input size error")));
//         } else if (!inputs[0].isTensor() || !inputs[1].isShapeVariable()) {
//             return Err(InferError(ERROR_MSG("Edge type not support")));
//         } else {
//             auto input = inputs[0].tensor();
//             auto shape = inputs[1].shapeVariable().shape;
//             int pos_1 = -1;
//             Shape ans(shape.size());
//             for (int i = 0; i < shape.size(); ++i) {
//                 switch (shape[i]) {
//                     case -1:
//                         if (pos_1 >= 0) {
//                             return Err(InferError(ERROR_MSG("Invalid shape variable")));
//                         }
//                         pos_1 = i;
//                         ans[i] = 1;
//                         break;
//                     case 0:
//                         if (i >= input.shape.size()) {
//                             return Err(InferError(ERROR_MSG("Invalid shape variable")));
//                         }
//                         ans[i] = input.shape[i];
//                         break;
//                     default:
//                         ans[i] = shape[i];
//                         break;
//                 }
//             }
//             auto old = std::accumulate(input.shape.begin(), input.shape.end(), 1, std::multiplies<>());
//             auto now = std::accumulate(ans.begin(), ans.end(), 1, std::multiplies<>());
//             if (pos_1 != -1) {
//                 if (old % now != 0) {
//                     return Err(InferError(ERROR_MSG("Invalid shape variable")));
//                 } else {
//                     ans[pos_1] = old / now;
//                 }
//             } else if (old != now) {
//                 return Err(InferError(ERROR_MSG("Invalid shape variable")));
//             }
//             return Ok(Edges{EdgeInfo{Tensor{input.dataType, std::move(ans)}}});
//         }
//     }

//     InferResult inferBatchNormalization(Edges inputs, bool training) {
//         if (inputs.size() != 5) {
//             return Err(InferError(ERROR_MSG("Input size error")));
//         } else if (std::any_of(inputs.begin(), inputs.end(), [](auto const &edge) { return !edge.isTensor(); })) {
//             return Err(InferError(ERROR_MSG("Edge type not support")));
//         } else {
//             auto input = inputs[0].tensor();
//             auto scale = inputs[1].tensor();
//             auto bias = inputs[2].tensor();
//             auto mean = inputs[3].tensor();
//             auto variance = inputs[4].tensor();
//             DataType dt[]{input.dataType, scale.dataType, mean.dataType};
//             if (!std::all_of(dt, dt + 3, isFloatDataType) || bias.dataType != dt[1] || variance.dataType != dt[2]) {
//                 return Err(InferError(ERROR_MSG("Input data type not support")));
//             }
//             if (input.shape.size() < 1 || scale.shape != bias.shape || scale.shape != mean.shape || scale.shape != variance.shape) {
//                 return Err(InferError(ERROR_MSG("Input shape not support")));
//             }
//             if (!training) {
//                 return Ok(Edges{std::move(input)});
//             } else {
//                 return Err(InferError(ERROR_MSG("Not implemented")));
//             }
//         }
//     }
