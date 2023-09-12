#include "infer.h"
#include <numeric>
#include <unordered_set>
#include <vector>

using namespace refactor::common;

namespace refactor::computation {

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
        : value(FatalError{}),
          std::runtime_error(std::forward<std::string>(msg)) {}
    InferError::InferError(UnknownVariable &&variable)
        : std::runtime_error(fmt::format("Unknown variable: {}", variable.name)),
          value(std::forward<UnknownVariable>(variable)) {}

#define EXPECT_SIZE(N)                                         \
    if (inputs.size() != (N)) {                                \
        return Err(InferError(ERROR_MSG("Input size error"))); \
    } else

#define EXPECT_VAL(DIM, VAL)                                             \
    int64_t VAL;                                                         \
    if ((DIM).hasValue()) {                                              \
        VAL = (DIM).value();                                             \
    } else {                                                             \
        return Err(InferError(UnknownVariable{(DIM.variable()->name)})); \
    }

    InferResult inferUnary(Operator const &op, Edges inputs) {
        EXPECT_SIZE(1) {
            auto dataType = inputs[0]->dataType;
            auto name = op.opType.name();
            static std::unordered_set<std::string> const SET[]{
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

    InferResult inferArithmetic(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2) {
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

    InferResult inferCumSum(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2)
        if (!inputs[1]->shape.empty()) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        } else if (!isNumbericDataType(inputs[0]->dataType) ||
                   (inputs[1]->dataType != DataType::I64 &&
                    inputs[1]->dataType != DataType::I32)) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        } else {
            return Ok(Edges{std::move(inputs[0])});
        }
    }

    InferResult inferSlice(Operator const &op, Edges inputs) {
        if (inputs.size() < 3 || 5 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            auto const &data = inputs[0];
            auto const &starts_ = inputs[1];
            auto const &ends_ = inputs[2];
            auto tint = starts_->dataType;
            if (tint != DataType::I32 && tint != DataType::I64) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
            if (ends_->dataType != tint) {
                return Err(InferError(ERROR_MSG("Data type not support")));
            }
            if (starts_->shape.size() != 1) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            if (ends_->shape != starts_->shape) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            EXPECT_VAL(starts_->shape[0], rank)
            if (data->shape.size() != rank) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            if (!starts_->hasData() || !ends_->hasData()) {
                return Err(InferError(ERROR_MSG("Starts and ends must be constant")));
            }
            int64_t *starts = reinterpret_cast<int64_t *>(starts_->data->ptr),
                    *ends = reinterpret_cast<int64_t *>(ends_->data->ptr),
                    *axes = nullptr,
                    *steps = nullptr;
            std::vector<int64_t> axes__, steps__;
            if (inputs.size() < 4) {
                axes__.resize(rank);
                axes = axes__.data();
                for (int64_t i = 0; i < rank; ++i) {
                    axes[i] = i;
                }
            } else {
                auto const &axes_ = inputs[3];
                if (axes_->dataType != tint || axes_->shape != starts_->shape) {
                    return Err(InferError(ERROR_MSG("Axes not support")));
                }
                if (!axes_->hasData()) {
                    return Err(InferError(ERROR_MSG("Axes must be constant")));
                }
                axes = reinterpret_cast<int64_t *>(axes_->data->ptr);
            }
            if (inputs.size() < 5) {
                steps__.resize(rank, 1);
                steps = steps__.data();
            } else {
                auto const &steps_ = inputs[4];
                if (steps_->dataType != tint || steps_->shape != starts_->shape) {
                    return Err(InferError(ERROR_MSG("Steps not support")));
                }
                if (!steps_->hasData()) {
                    return Err(InferError(ERROR_MSG("Steps must be constant")));
                }
                steps = reinterpret_cast<int64_t *>(steps_->data->ptr);
            }

            Shape ans(rank, DimExpr(1));
            for (size_t i = 0; i < rank; ++i) {
                auto axis = axes[i];
                auto start = starts[i];
                auto end = ends[i];
                auto step = steps[i];
                if (axis < 0) {
                    axis += rank;
                }
                EXPECT_VAL(data->shape[axis], dim)
                if (start < 0) {
                    start += dim;
                }
                if (end < 0) {
                    end += dim;
                }
                if (start < 0 || dim <= start || end < 0 || dim < end) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                if (step > 0) {
                    ans[axis] = DimExpr((end - start + step - 1) / step);
                } else if (step < 0) {
                    ans[axis] = DimExpr((end - start - step + 1) / -step);
                } else {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
            }
            return Ok(Edges{std::make_shared<Tensor>(data->dataType, std::move(ans))});
        }
    }

    InferResult inferShape(Operator const &op, Edges inputs) {
        EXPECT_SIZE(1) {
            auto attrs = op.attributes;
            auto start = op.attribute("start", {0}).int_(),
                 end = op.attribute("end", {-1}).int_();

            ASSERT(start == 0, "only support start == 0");
            ASSERT(end == -1, "only support end == -1");

            auto ans = inputs[0]->shape;
            auto blob = std::make_shared<Blob>(new int64_t[ans.size()]);
            for (auto i = 0; i < ans.size(); ++i) {
                EXPECT_VAL(ans[i], v)
                reinterpret_cast<int64_t *>(blob->ptr)[i] = v;
            }
            return Ok(Edges{
                std::make_shared<Tensor>(
                    DataType::I64,
                    Shape{DimExpr(ans.size())},
                    std::move(blob)),
            });
        }
    }

    InferResult inferWhere(Operator const &op, Edges inputs) {
        EXPECT_SIZE(3) {
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

    InferResult inferSqueeze(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2) {
            auto const &data = inputs[0];
            auto const &axes = inputs[1];
            if (axes->dataType != DataType::I64 || axes->shape.size() != 1) {
                return Err(InferError(ERROR_MSG("Axes not support")));
            }
            auto axes_ = reinterpret_cast<int64_t *>(axes->data->ptr);
            EXPECT_VAL(axes->shape[0], axesSize)
            std::vector<int64_t> axes__(axes_, axes_ + axesSize);
            for (auto &i : axes__) {
                if (i < 0) {
                    i += data->shape.size();
                }
            }
            std::sort(axes__.begin(), axes__.end());
            Shape ans;
            if (std::strcmp(op.opType.name(), "onnx::Squeeze") == 0) {
                auto len = data->shape.size();
                auto itx = data->shape.begin();
                auto ity = axes__.begin();
                ans = Shape(len, DimExpr(1));
                for (auto i = 0; i < len; ++i) {
                    if (i != *ity) {
                        ans[i] = *itx++;
                    } else {
                        ASSERT(*itx++ == DimExpr(1), "Unsqueeze error");
                        ity++;
                    }
                }
            } else if (std::strcmp(op.opType.name(), "onnx::Unsqueeze") == 0) {
                auto len = data->shape.size() + axes__.size();
                auto itx = data->shape.begin();
                auto ity = axes__.begin();
                ans = Shape(len, DimExpr(1));
                for (size_t i = 0; i < len; ++i) {
                    if (i != *ity) {
                        ans[i] = *itx++;
                    } else {
                        ans[i] = DimExpr(1);
                        ity++;
                    }
                }
            } else {
                RUNTIME_ERROR(fmt::format("OpType {} not support in squeeze inference", op.opType.name()));
            }
            return Ok(Edges{std::make_shared<Tensor>(data->dataType, std::move(ans))});
        }
    }

    InferResult inferEqual(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2) {
            auto const &a = inputs[0];
            auto const &b = inputs[1];
            if (a->dataType != b->dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            auto ans = multidirBroadcast({a->shape, b->shape});
            if (ans.isErr()) {
                return Err(InferError(ERROR_MSG(ans.unwrapErr())));
            } else {
                return Ok(Edges{std::make_shared<Tensor>(DataType::Bool, ans.unwrap())});
            }
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

    InferResult inferConcat(Operator const &op, Edges inputs) {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        } else {
            auto shape = inputs[0]->shape;
            auto dataType = inputs[0]->dataType;
            auto axis = op.attribute("axis").int_();
            for (auto it = inputs.begin() + 1; it != inputs.end(); ++it) {
                auto const &input = *it;
                if (input->dataType != dataType) {
                    return Err(InferError(ERROR_MSG("Input data type not support")));
                }
                if (input->shape.size() != shape.size()) {
                    return Err(InferError(ERROR_MSG("Input shape not support")));
                }
                for (size_t i = 0; i < shape.size(); ++i) {
                    if (i == axis) {
                        EXPECT_VAL(shape[i], a)
                        EXPECT_VAL(input->shape[i], b)
                        shape[i] = DimExpr(a + b);
                    } else if (shape[i] != input->shape[i]) {
                        return Err(InferError(ERROR_MSG("Input shape not support")));
                    }
                }
            }
            return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(shape))});
        }
    }

    InferResult inferGather(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2)
        if (inputs[1]->dataType != DataType::I32 && inputs[1]->dataType != DataType::I64) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        } else {
            auto const r = inputs[0]->shape.size();
            auto const q = inputs[1]->shape.size();
            auto axis = op.attribute("axis", {0}).int_();
            if (axis < 0) {
                axis += r;
            }
            if (axis < 0 || r <= axis) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            auto dataType = inputs[0]->dataType;
            auto ans = inputs[0]->shape;
            auto const &indices = inputs[1]->shape;
            ans.erase(ans.begin() + axis);
            ans.insert(ans.begin() + axis, indices.begin(), indices.end());
            if (inputs[0]->hasData() && std::all_of(ans.begin(), ans.end(), [](auto const &d) { return d.hasValue(); })) {
                auto const ssz = ans.size();
                std::vector<int64_t> sX(r), sY(q), sZ(ssz);

                auto getter = [](DimExpr const &d) { return d.value(); };
                std::transform(inputs[0]->shape.begin(), inputs[0]->shape.end(), sX.begin(), getter);
                std::transform(inputs[1]->shape.begin(), inputs[1]->shape.end(), sY.begin(), getter);
                std::transform(ans.begin(), ans.end(), sZ.begin(), getter);

                auto size = std::accumulate(sZ.begin(), sZ.end(), 1, std::multiplies());
                auto eleSize = dataTypeSize(inputs[0]->dataType);
                auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
                auto srcX = reinterpret_cast<uint8_t *>(inputs[0]->data->ptr);
                auto srcY = reinterpret_cast<int64_t *>(inputs[1]->data->ptr);
                auto dst = reinterpret_cast<uint8_t *>(blob->ptr);

                for (size_t i = 0; i < size; ++i) {
                    std::vector<int64_t> indices(ssz);
                    auto zi = i;
                    for (size_t j = 0; j < ssz; ++j) {
                        auto div = std::div(zi, sZ[j]);
                        indices[ssz - 1 - j] = div.rem;// reverse
                        zi = div.quot;
                    }

                    size_t ii = 0, mul = 1;
                    for (size_t j = 0; j < q; ++j) {
                        auto j_ = q - 1 - j;// reverse
                        ii += indices[j_] * mul;
                        mul *= sY[j_];
                    }
                    auto k = srcY[ii];

                    ii = 0, mul = 1;
                    for (size_t j = axis + q; j < ssz; ++j) {
                        auto j_ = ssz - 1 - j;// reverse
                        ii += indices[j_] * mul;
                        mul *= sX[j_ - q + 1];
                    }
                    ii += k * mul;
                    for (size_t j = 0; j < axis; ++j) {
                        auto j_ = axis - 1 - j;// reverse
                        ii += indices[j_] * mul;
                        mul *= sX[j_];
                    }

                    std::copy_n(srcX + ii * eleSize, eleSize, dst + i * eleSize);
                }

                return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(ans), std::move(blob))});
            } else {
                return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(ans))});
            }
        }
    }

    InferResult inferCast(Operator const &op, Edges inputs) {
        EXPECT_SIZE(1) {
            auto to = static_cast<DataType>(op.attribute("to").int_());
            return Ok(Edges{std::make_shared<Tensor>(to, inputs[0]->shape)});
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

    InferResult inferExpand(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2)
        if (inputs[1]->dataType != DataType::I64 ||
            inputs[1]->shape.size() != 1 ||
            !inputs[1]->hasData()) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        } else {
            auto const &data = inputs[0];
            auto const &shape = inputs[1];
            auto shape_ = reinterpret_cast<int64_t *>(shape->data->ptr);
            EXPECT_VAL(shape->shape[0], shapeSize)
            Shape shape__(shape_, shape_ + shapeSize);
            auto res = multidirBroadcast({data->shape, shape__});
            if (res.isErr()) {
                return Err(InferError(ERROR_MSG(res.unwrapErr())));
            } else {
                return Ok(Edges{std::make_shared<Tensor>(data->dataType, res.unwrap())});
            }
        }
    }

    InferResult inferConstantOfShape(Operator const &op, Edges inputs) {
        EXPECT_SIZE(1)
        if (auto shape = inputs[0];
            shape->dataType != DataType::I64 ||
            shape->shape.size() != 1 ||
            !shape->hasData()) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        } else {
            auto shape_ = reinterpret_cast<int64_t *>(shape->data->ptr);
            EXPECT_VAL(inputs[0]->shape[0], shapeSize)
            Shape ans(shapeSize, DimExpr(1));
            size_t size = 1;
            for (auto i = 0; i < shapeSize; ++i) {
                auto d = shape_[i];
                ans[i] = DimExpr(d);
                size *= d;
            }
            if (auto it = op.attributes.find("value"); it != op.attributes.end()) {
                auto const &value = it->second.tensor();
                ASSERT(value->hasData(), "ConstantOfShape value must have data");
                ASSERT(value->shape.size() == 1 && value->shape[0] == DimExpr(1), "ConstantOfShape value must be scalar");
                auto dataType = value->dataType;
                auto eleSize = dataTypeSize(dataType);
                auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
                for (auto i = 0; i < size; ++i) {
                    std::memcpy(reinterpret_cast<uint8_t *>(blob->ptr) + i * eleSize, value->data->ptr, eleSize);
                }
                return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(ans), std::move(blob))});
            } else {
                auto dataType = DataType::F32;
                auto eleSize = dataTypeSize(dataType);
                auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
                std::memset(blob->ptr, 0, size * eleSize);
                return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(ans), std::move(blob))});
            }
        }
    }

}// namespace refactor::computation

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
