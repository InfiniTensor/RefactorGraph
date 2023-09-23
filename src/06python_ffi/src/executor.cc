#include "executor.h"

namespace refactor::python_ffi {

    Executor::Executor(computation::Graph g)
        : _g(std::move(g)) {}

}// namespace refactor::python_ffi
