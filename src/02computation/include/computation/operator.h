#ifndef COMPUTATION_OPERATOR_H
#define COMPUTATION_OPERATOR_H

#include <memory>
#include <string>

namespace refactor::computation {

    class Operator {
    public:
    };

    using SharedOp = std::shared_ptr<Operator>;

    struct Node {
        SharedOp op;
        std::string name;
    };

}// namespace refactor::computation

#endif// COMPUTATION_OPERATOR_H
