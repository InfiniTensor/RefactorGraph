#ifndef COMPUTATION_OPERATOR_H
#define COMPUTATION_OPERATOR_H

#include <memory>
#include <string>

namespace refactor::computation {

    class Operator {
    public:
        virtual size_t opTypeId() const = 0;
        virtual std::string_view name() const = 0;
        virtual bool isLayoutDependent() const { return false; }
    };

    using SharedOp = std::shared_ptr<Operator>;

    struct Node {
        SharedOp op;
        std::string name;
    };

}// namespace refactor::computation

#endif// COMPUTATION_OPERATOR_H
