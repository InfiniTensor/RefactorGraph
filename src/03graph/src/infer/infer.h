#ifndef INFER_H
#define INFER_H

#include "graph/edge_info.h"
#include <vector>

namespace refactor::graph {

    std::vector<EdgeInfo> inferAbs(std::vector<EdgeInfo> inputs);
    std::vector<EdgeInfo> inferTrigonometry(std::vector<EdgeInfo> inputs);
    std::vector<EdgeInfo> inferTanh(std::vector<EdgeInfo> inputs);
    std::vector<EdgeInfo> inferArithmetic(std::vector<EdgeInfo> inputs);

}// namespace refactor::graph

#endif// INFER_H
