#ifndef COMPUTATION_CONCAT_H
#define COMPUTATION_CONCAT_H

#include "../operator.h"

namespace refactor::computation {

    struct Concat : public Operator {
        size_t axis;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CONCAT_H
