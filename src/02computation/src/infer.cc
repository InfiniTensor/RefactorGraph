#include "computation/infer.h"

namespace refactor::computation {

    InferError::InferError(std::string msg)
        : value(FatalError{}),
          std::runtime_error(std::forward<std::string>(msg)) {}
    InferError::InferError(UnknownVariable variable)
        : std::runtime_error(fmt::format("Unknown variable: {}", variable.name)),
          value(std::forward<UnknownVariable>(variable)) {}

}// namespace refactor::computation
