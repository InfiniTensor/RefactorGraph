#include "executor.h"

namespace refactor::python_ffi {

    Executor::Executor(computation::Graph g, kernel::Target t)
        : _g(g.lower(t)) {
        _g.transpose();
    }

}// namespace refactor::python_ffi
