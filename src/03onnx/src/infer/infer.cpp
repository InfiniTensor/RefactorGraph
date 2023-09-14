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

    bool shouldCalculate(Tensors const &inputs, Shape const &output) {
        return std::all_of(inputs.begin(), inputs.end(), [](auto const &input) { return input->hasData(); }) &&
               std::all_of(output.begin(), output.end(), [](auto const &dim) { return dim.hasValue(); });
    }

    size_t sizeOf(Shape const &shape) {
        return std::accumulate(shape.begin(), shape.end(), 1,
                               [](auto acc, const auto &d) { return acc * d.value(); });
    }

    Indices buildIndices(Shape const &shape, size_t i) {
        Indices indices(shape.size());
        auto it = indices.rbegin();
        for (auto d : shape) {
            auto div = std::div(i, d.value());
            *it++ = div.rem;
            i = div.quot;
        }
        return indices;
    }

    void *locate(Tensor const &tensor, Indices const &indices) {
        auto it0 = indices.rbegin(),
             end0 = indices.rend();
        auto it1 = tensor.shape.rbegin(),
             end1 = tensor.shape.rend();
        size_t i = 0, mul = 1;
        while (it0 != end0 && it1 != end1) {
            i += *it0++ * mul;
            mul *= it1++->value();
        }
        return reinterpret_cast<uint8_t *>(tensor.data->ptr) + i * dataTypeSize(tensor.dataType);
    }
}// namespace refactor::onnx

//     InferResult inferConv(Tensors inputs, ShapeOrNot dilations, ShapeOrNot pads, ShapeOrNot strides) {
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
//                 return Ok(Tensors{EdgeInfo{Tensor{dataType, std::move(ans)}}});
//             }
//         }
//     }

//     InferResult inferPool(Tensors inputs, ShapeOrNot dilations, Shape kernelShape, ShapeOrNot pads, ShapeOrNot strides) {
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
//                 return Ok(Tensors{EdgeInfo{Tensor{dataType, std::move(ans)}}});
//             }
//         }
//     }

//     InferResult inferGlobalPool(Tensors inputs) {
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
//             return Ok(Tensors{EdgeInfo{Tensor{input.dataType, std::move(ans)}}});
//         }
//     }

//     InferResult inferBatchNormalization(Tensors inputs, bool training) {
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
//                 return Ok(Tensors{std::move(input)});
//             } else {
//                 return Err(InferError(ERROR_MSG("Not implemented")));
//             }
//         }
//     }
