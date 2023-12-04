#include "extra_padding.cuh"

namespace refactor::kernel {

    std::optional<ExtraPadding>
    ExtraPadding::build(DataType dt, int const *shape, int const *pads) {
        if (pads[0] == pads[2] && pads[1] == pads[3]) {
            return std::nullopt;
        }
        int padH = pads[0] - pads[2], padW = pads[1] - pads[3];
        return ExtraPadding{
            dt,
            shape[0] * shape[1],
            (shape[2] + std::abs(padH)) * (shape[3] + std::abs(padW)),
            shape[3] + std::abs(padW),
            shape[2],
            shape[3],
            padH,
            padW};
    }

    size_t
    ExtraPadding::workspace() const {
        return nc * sohw * dt.size();
    }


    void const *
    ExtraPadding::operator()(void const *src, void *workspace_) const {
        auto extra = reinterpret_cast<uint8_t *>(workspace_);

#define CASE(T)                                                       \
    case DataType::T: {                                               \
        using T_ = primitive<DataType::T>::type;                      \
        thrust::tabulate(thrust::device,                              \
                         reinterpret_cast<T_ *>(extra),               \
                         reinterpret_cast<T_ *>(extra + workspace()), \
                         ExtraPaddingFunctor<T_>{*this, src});        \
    } break;

        switch (dt) {
            CASE(F32)
            CASE(F64)
            default:
                UNREACHABLE();
        }
        return workspace_;
    }

}// namespace refactor::kernel
