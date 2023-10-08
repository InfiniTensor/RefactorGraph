#include "executor.h"

namespace refactor::python_ffi {

    Executor::Executor(computation::Graph g)
        : _g(std::move(g)) {
        _g.transpose();
    }

}// namespace refactor::python_ffi
