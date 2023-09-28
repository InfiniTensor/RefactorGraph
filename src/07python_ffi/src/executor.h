#ifndef PYTHON_FFI_EXECUTOR_H
#define PYTHON_FFI_EXECUTOR_H

#include "computation/graph.h"

namespace refactor::python_ffi {

    class Executor {
        computation::Graph _g;

    public:
        explicit Executor(computation::Graph);
    };

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_EXECUTOR_H
