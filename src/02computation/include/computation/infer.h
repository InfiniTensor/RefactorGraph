#ifndef COMPUTATION_INFER_H
#define COMPUTATION_INFER_H

#include "tensor.h"
#include <result.h>

namespace refactor::computation {

    class Operator;
    using Edges = std::vector<Edge>;
    using Tensors = std::vector<std::shared_ptr<Tensor>>;

    struct FatalError {};
    struct UnknownVariable {
        std::string name;
    };
    struct InferError : public std::runtime_error {
        std::variant<FatalError, UnknownVariable> value;

        explicit InferError(std::string);
        explicit InferError(UnknownVariable);
    };
    using InferResult = Result<Tensors, InferError>;
    using InferFn = InferResult (*)(Operator const &, Tensors);

}// namespace refactor::computation

#endif// COMPUTATION_INFER_H
