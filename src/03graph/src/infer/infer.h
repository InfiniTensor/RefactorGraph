#ifndef INFER_H
#define INFER_H

#include "graph/edge_info.h"
#include <result.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace refactor::graph {

    using Edges = std::vector<EdgeInfo>;
    struct InferError : public std::runtime_error {
        explicit InferError(std::string &&msg);
    };
    using InferResult = Result<Edges, InferError>;

#define INFER_ERROR(msg) InferError(buildMsg(msg, __FILE__, __LINE__))

    InferResult inferAbs(Edges inputs);
    InferResult inferTrigonometry(Edges inputs);
    InferResult inferTanh(Edges inputs);
    InferResult inferArithmetic(Edges inputs);

}// namespace refactor::graph

#endif// INFER_H
