#ifndef PYTHON_FFI_EXECUTOR_H
#define PYTHON_FFI_EXECUTOR_H

#include "functions.h"
#include "kernel/graph.h"

namespace refactor::python_ffi {
    using SharedTensor = Arc<frontend::Tensor>;

    class Executor {
        kernel::Graph _graph;
        kernel::Allocator _allocator;
        runtime::Stream _stream;

    public:
        Executor(kernel::Graph, kernel::Allocator);
        void setInput(uint_lv1, pybind11::array);
        std::vector<uint_lv1> prepare();
    };

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_EXECUTOR_H
