#ifndef PYTHON_FFI_EXECUTOR_H
#define PYTHON_FFI_EXECUTOR_H

#include "runtime/stream.h"

namespace refactor::python_ffi {

    class Executor {
        runtime::Stream _stream;

    public:
        explicit Executor(runtime::Stream);
    };

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_EXECUTOR_H
