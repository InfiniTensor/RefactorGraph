#include "infer.h"
#include <numeric>
#include <unordered_set>
#include <vector>

namespace refactor::onnx {
    using namespace refactor::common;

    ShapeResult multidirBroadcast(std::vector<Shape> const &inputs) {
        using Iter = std::reverse_iterator<Shape::const_iterator>;
        std::vector<std::pair<Iter, Iter>> iters;
        iters.reserve(inputs.size());
        for (auto const &input : inputs) {
            iters.emplace_back(input.rbegin(), input.rend());
        }
        Shape ans;
        while (true) {
            std::optional<DimExpr> dim = std::nullopt;
            for (size_t i = 0; i < iters.size();) {
                if (iters[i].first != iters[i].second) {
                    auto new_ = *iters[i].first++;
                    if (!dim || *dim == DimExpr(1)) {
                        dim = std::move(new_);
                    } else if (new_ != DimExpr(1) && new_ != *dim) {
                        fmt::print("Shape broadcast failed: ");
                        for (auto const &input : inputs) {
                            fmt::print("{} ", shapeFormat(input));
                        }
                        fmt::println("");
                        return Err(ERROR_MSG("Shape broadcast failed"));
                    }
                    ++i;
                } else {
                    std::swap(iters[i], iters.back());
                    iters.pop_back();
                }
            }
            if (dim) {
                ans.emplace_back(std::move(*dim));
            } else {
                break;
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

    bool shouldCalculate(Edges const &inputs, Shape const &output) {
        return std::all_of(inputs.begin(), inputs.end(), [](auto const &input) { return input->hasData(); }) &&
               std::all_of(output.begin(), output.end(), [](auto const &dim) { return dim.hasValue(); });
    }

    std::pair<absl::InlinedVector<int64_t, 4>, size_t> shape_size(Shape const &shape) {
        absl::InlinedVector<int64_t, 4> ans;
        size_t size = 1;
        ans.reserve(shape.size());
        for (auto const &d : shape) {
            auto value = d.value();
            ans.push_back(value);
            size *= value;
        }
        return {std::move(ans), size};
    }

    absl::InlinedVector<int64_t, 4> buildIndices(absl::InlinedVector<int64_t, 4> const &shape, size_t i) {
        absl::InlinedVector<int64_t, 4> indices(shape.size());
        auto it = indices.rbegin();
        for (auto d : shape) {
            auto div = std::div(i, d);
            *it++ = div.rem;
            i = div.quot;
        }
        return indices;
    }

    InferResult inferUnary(Operator const &op, Edges inputs) {
        EXPECT_SIZE(1) {
            auto dataType = inputs[0]->dataType;
            auto name = op.opType.name();
            static std::unordered_set<std::string_view> const SET[]{
                {"onnx::Abs", "onnx::Relu", "onnx::PRelu"},
                {"onnx::Acos", "onnx::Acosh",
                 "onnx::Asin", "onnx::Asinh",
                 "onnx::Atan", "onnx::Atanh",
                 "onnx::Cos", "onnx::Cosh",
                 "onnx::Sin", "onnx::Sinh",
                 "onnx::Tan"},
                {"onnx::Tanh", "onnx::Sqrt"}};
            if (SET[0].find(name) != SET[0].end()) {
                if (!isNumbericDataType(dataType)) {
                    return Err(InferError(ERROR_MSG("Data type not support")));
                }
            } else if (SET[1].find(name) != SET[1].end()) {
                if (!isIeee754DataType(dataType)) {
                    return Err(InferError(ERROR_MSG("Data type not support")));
                }
            } else if (SET[2].find(name) != SET[2].end()) {
                if (!isFloatDataType(dataType)) {
                    return Err(InferError(ERROR_MSG("Data type not support")));
                }
            } else {
                RUNTIME_ERROR(fmt::format("OpType {} not support in unary inference", op.opType.name()));
            }
            return Ok(std::move(inputs));
        }
    }

    InferResult inferGemm(Operator const &op, Edges inputs) {
        if (auto size = inputs.size(); size < 2 || 3 < size) {
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

            EXPECT_VAL(a->shape[0], a0)
            EXPECT_VAL(a->shape[1], a1)
            EXPECT_VAL(b->shape[0], b0)
            EXPECT_VAL(b->shape[1], b1)

            size_t m, n, k;
            if (op.attribute("transA", {0}).int_() == 0) {
                m = a0;
                k = a1;
            } else {
                m = a1;
                k = a0;
            }
            if (op.attribute("transB", {0}).int_() == 0) {
                if (b0 != k) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                n = b1;
            } else {
                if (b1 != k) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                n = b0;
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

    InferResult inferMatMul(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2) {
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

    InferResult inferReshape(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2)
        if (auto shape = inputs[1];
            shape->dataType != DataType::I64 ||
            shape->shape.size() != 1 ||
            !shape->hasData()) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        } else {
            auto const &input = inputs[0]->shape;
            auto shapeValue = reinterpret_cast<int64_t *>(inputs[1]->data->ptr);
            EXPECT_VAL(inputs[1]->shape[0], rank)
            Shape ans(rank, DimExpr(1));
            auto pos_1 = -1;
            for (auto i = 0; i < ans.size(); ++i) {
                switch (shapeValue[i]) {
                    case -1:
                        if (pos_1 != -1) {
                            return Err(InferError(ERROR_MSG("Invalid shape variable")));
                        }
                        pos_1 = i;
                        break;
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
            size_t old_ = 1, new_ = 1;
            for (auto const &d : input) {
                EXPECT_VAL(d, v)
                old_ *= v;
            }
            for (auto const &d : ans) {
                EXPECT_VAL(d, v)
                new_ *= v;
            }
            if (pos_1 != -1) {
                if (old_ % new_ != 0) {
                    return Err(InferError(ERROR_MSG("Invalid shape variable")));
                } else {
                    ans[pos_1] = DimExpr(old_ / new_);
                }
            } else if (old_ != new_) {
                return Err(InferError(ERROR_MSG("Invalid shape variable")));
            }
            return Ok(Edges{std::make_shared<Tensor>(inputs[0]->dataType, std::move(ans))});
        }
    }

    InferResult inferSoftmax(Operator const &op, Edges inputs) {
        EXPECT_SIZE(1)
        if (!isIeee754DataType(inputs[0]->dataType)) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        } else {
            return Ok(std::move(inputs));
        }
    }

    InferResult inferPow(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2) {
            auto const &a = inputs[0];
            auto const &b = inputs[1];
            if (!isSignedDataType(a->dataType) || !isNumbericDataType(b->dataType)) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            auto ans = multidirBroadcast({a->shape, b->shape});
            if (ans.isErr()) {
                return Err(InferError(ERROR_MSG(ans.unwrapErr())));
            } else {
                return Ok(Edges{std::make_shared<Tensor>(a->dataType, ans.unwrap())});
            }
        }
    }

    InferResult inferReduce(Operator const &op, Edges inputs) {
        if (inputs.empty() || 2 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else if (!isNumbericDataType(inputs[0]->dataType)) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        } else {
            auto keepdims = op.attribute("keepdims", {1}).int_();
            if (inputs.size() == 2) {
                auto const &shape = inputs[0]->shape;
                auto const &axes = inputs[1];
                if (axes->dataType != DataType::I64 ||
                    axes->shape.size() != 1 ||
                    !axes->hasData()) {
                    return Err(InferError(ERROR_MSG("Axes not support")));
                }
                auto axes_ = reinterpret_cast<int64_t *>(axes->data->ptr);
                EXPECT_VAL(axes->shape[0], axesSize)
                std::unordered_set<int64_t> axes__;
                for (size_t i = 0; i < axesSize; ++i) {
                    auto axis = axes_[i];
                    axes__.insert(axis < 0 ? axis + shape.size() : axis);
                }
                Shape ans;
                for (size_t i = 0; i < shape.size(); ++i) {
                    if (axes__.find(i) == axes__.end()) {
                        ans.emplace_back(shape[i]);
                    } else if (keepdims) {
                        ans.emplace_back(1);
                    }
                }
                return Ok(Edges{std::make_shared<Tensor>(inputs[0]->dataType, std::move(ans))});
            } else if (op.attribute("noop_with_empty_axes", {0}).int_() != 0) {
                return Ok(Edges{std::move(inputs[0])});
            } else if (keepdims) {
                return Ok(Edges{std::make_shared<Tensor>(inputs[0]->dataType, Shape(inputs[0]->shape.size(), DimExpr(1)))});
            } else {
                return Ok(Edges{std::make_shared<Tensor>(inputs[0]->dataType, Shape{})});
            }
        }
    }

    InferResult inferMax(Operator const &op, Edges inputs) {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            std::vector<Shape> shapes;
            shapes.reserve(inputs.size());
            auto dataType = inputs[0]->dataType;
            for (auto &input : inputs) {
                if (input->dataType != dataType) {
                    return Err(InferError(ERROR_MSG("Input data type not support")));
                }
                shapes.emplace_back(std::move(input->shape));
            }
            auto shape = multidirBroadcast(shapes);
            if (shape.isErr()) {
                return Err(InferError(ERROR_MSG(shape.unwrapErr())));
            } else {
                return Ok(Edges{std::make_shared<Tensor>(dataType, shape.unwrap())});
            }
        }
    }

    InferResult inferTranspose(Operator const &op, Edges inputs) {
        EXPECT_SIZE(1) {
            auto const &data = inputs[0];
            auto const &attrs = op.attributes;
            if (auto it = attrs.find("perm"); it != attrs.end()) {
                auto const &perm = it->second.ints();
                if (perm.size() != data->shape.size()) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                Shape ans;
                for (auto i : perm) {
                    ans.emplace_back(data->shape[i]);
                }
                return Ok(Edges{std::make_shared<Tensor>(data->dataType, std::move(ans))});
            } else {
                Shape ans(data->shape.rbegin(), data->shape.rend());
                return Ok(Edges{std::make_shared<Tensor>(data->dataType, std::move(ans))});
            }
        }
    }
}// namespace refactor::onnx

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
