#include "kernel/attributes/mat_mul_integer_info.h"

namespace refactor::kernel {

#define A (inputs[0].get().shape)
#define B (inputs[1].get().shape)

    MatMulIntegerInfo::Input::Input(TensorRefs const &inputs, size_t i) noexcept
        : signed_(inputs[i].get().dataType == DataType::I8),
          withZeroPoint(false) {
        if (inputs.size() > i + 2) {
            auto const &t = inputs[i + 2].get();
            withZeroPoint = t.rank() != 0 || !t.data || t.data->get<uint8_t>() != 0;
        }
    }

    MatMulIntegerInfo::MatMulIntegerInfo(TensorRefs const &inputs) noexcept
        : a(inputs, 0),
          b(inputs, 1),
          m(A.rbegin()[1]),
          k(A.rbegin()[0]),
          n(B.rbegin()[0]),
          broadcaster({slice(A.data(), A.size() - 2),
                       slice(B.data(), B.size() - 2)}) {}

}// namespace refactor::kernel
