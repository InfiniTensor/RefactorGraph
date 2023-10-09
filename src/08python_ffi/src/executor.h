#ifndef PYTHON_FFI_EXECUTOR_H
#define PYTHON_FFI_EXECUTOR_H

#include "computation/graph.h"
#include "kernel/target.h"

namespace refactor::python_ffi {

    class Executor {
        kernel::Graph _g;

    public:
        explicit Executor(computation::Graph, kernel::Target);
    };

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_EXECUTOR_H
