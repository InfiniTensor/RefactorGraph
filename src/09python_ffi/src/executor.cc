#include "executor.h"

namespace refactor::python_ffi {

    Executor::Executor(runtime::Stream stream)
        : _stream(std::move(stream)) {
    }

}// namespace refactor::python_ffi
