#include "infer.h"
#include "common/op_type.h"
#include <numeric>
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
        } else {
            auto const &a = inputs[0];
            auto const &b = inputs[1];
            auto dataType = a->dataType;
            if (!isNumbericDataType(dataType) || b->dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (a->shape.size() != 2 || b->shape.size() != 2) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }

            size_t m, n, k;
            if (node.operator_().attribute("transA", {0}).int_() == 0) {
                m = a->shape[0].value();
                k = a->shape[1].value();
            } else {
                m = a->shape[1].value();
                k = a->shape[0].value();
            }
            if (node.operator_().attribute("transB", {0}).int_() == 0) {
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
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            auto const &a = inputs[0];
            auto const &b = inputs[1];
            auto dataType = a->dataType;
            if (!isNumbericDataType(dataType) || b->dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            auto sa = a->shape, sb = b->shape;
            switch (sa.size()) {
                case 1:
                    sa.insert(sa.begin(), DimExpr(1));
                    break;
                case 0:
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                default:
                    break;
            }
            switch (sb.size()) {
                case 1:
                    sb.emplace_back(1);
                    break;
                case 0:
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                default:
                    break;
            }
            auto k = *sa.rbegin();
            sa.pop_back();
            auto m = *sa.rbegin();
            sa.pop_back();
            auto n = *sb.rbegin();
            sb.pop_back();
            if (k != *sb.rbegin()) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            sb.pop_back();
            if (!sa.empty() || !sb.empty()) {
                auto res = multidirBroadcast({sa, sb});
                if (res.isErr()) {
                    return Err(InferError(ERROR_MSG(res.unwrapErr())));
                } else {
                    auto shape = res.unwrap();
                    shape.emplace_back(m);
                    shape.emplace_back(n);
                    return Ok(Edges{std::make_shared<Tensor>(dataType, shape)});
                }
            } else {
                return Ok(Edges{std::make_shared<Tensor>(dataType, Shape{m, n})});
            }
        }
    }

    InferResult inferReshape(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (!inputs[1]->hasData()) {
            return Err(InferError(ERROR_MSG("Shape must be constant")));
        } else if (inputs[1]->dataType != DataType::I64 || inputs[1]->shape.size() > 1) {
            return Err(InferError(ERROR_MSG("Shape must be 1D int64 tensor")));
        } else {
            auto const &input = inputs[0]->shape;
            auto shapeValue = reinterpret_cast<int64_t *>(inputs[1]->data->ptr);
            Shape ans(inputs[1]->shape[0].value(), DimExpr(1));
            auto pos_1 = -1;
            for (auto i = 0; i < ans.size(); ++i) {
                switch (shapeValue[i]) {
                    case -1:
                        if (pos_1 == -1) {
                            return Err(InferError(ERROR_MSG("Invalid shape variable")));
                        } else {
                            pos_1 = i;
                            break;
                        }
                    case 0:
                        if (i >= input.size()) {
                            return Err(InferError(ERROR_MSG("Invalid shape variable")));
                        }
                        ans[i] = input[i];
                        break;
                    default:
                        ans[i] = DimExpr(shapeValue[i]);
                        break;
                }
            }
            auto old = std::accumulate(input.begin(), input.end(), 1, [](auto const &acc, auto const &d) { return acc * d.value(); });
            auto now = std::accumulate(ans.begin(), ans.end(), 1, [](auto const &acc, auto const &d) { return acc * d.value(); });
            if (pos_1 != -1) {
                if (old % now != 0) {
                    return Err(InferError(ERROR_MSG("Invalid shape variable")));
                } else {
                    ans[pos_1] = DimExpr(old / now);
                }
            } else if (old != now) {
                return Err(InferError(ERROR_MSG("Invalid shape variable")));
            }
            return Ok(Edges{std::make_shared<Tensor>(inputs[0]->dataType, std::move(ans))});
        }
    }

    InferResult inferCumSum(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferSlice(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferShape(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 1) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            auto attrs = node.operator_().attributes;
            auto start = node.operator_().attribute("start", {0}).int_(),
                 end = node.operator_().attribute("end", {-1}).int_();

            ASSERT(start == 0, "only support start == 0");
            ASSERT(end == -1, "only support end == -1");

            auto ans = inputs[0]->shape;
            auto blob = std::make_shared<Blob>(new int64_t[ans.size()]);
            for (auto i = 0; i < ans.size(); ++i) {
                reinterpret_cast<int64_t *>(blob->ptr)[i] = ans[i].value();
            }
            return Ok(Edges{
                std::make_shared<Tensor>(
                    DataType::I64,
                    Shape{DimExpr(ans.size())},
                    std::move(blob)),
            });
        }
    }

    InferResult inferWhere(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 3) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            auto const &condition = inputs[0];
            auto const &x = inputs[1];
            auto const &y = inputs[2];
            if (condition->dataType != DataType::Bool) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (x->dataType != y->dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            auto ans = multidirBroadcast({condition->shape, x->shape, y->shape});
            if (ans.isErr()) {
                return Err(InferError(ERROR_MSG(ans.unwrapErr())));
            } else {
                return Ok(Edges{std::make_shared<Tensor>(x->dataType, ans.unwrap())});
            }
        }
    }

    InferResult inferSqueeze(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 1) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferEqual(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferSoftmax(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferPow(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferReduce(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferConcat(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferGather(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferCast(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferUnsqueeze(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 1) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferMax(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferTranspose(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
    }

    InferResult inferExpand(NodeInfo const &node, Edges inputs) {
        if (inputs.size() != 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            return Err(InferError(ERROR_MSG("Not implemented")));
        }
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
