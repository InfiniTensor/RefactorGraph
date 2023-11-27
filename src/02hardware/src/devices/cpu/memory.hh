#ifndef HARDWARE_DEVICES_CPU_MEMORY_HH
#define HARDWARE_DEVICES_CPU_MEMORY_HH

#include "hardware/memory.h"

namespace refactor::hardware {

    class CpuMemory final : public Memory {
        void *malloc(size_t) noexcept final;
        void free(void *) noexcept final;
        void *copyHD(void *dst, void const *src, size_t bytes) const noexcept final;
        void *copyDH(void *dst, void const *src, size_t bytes) const noexcept final;
        void *copyDD(void *dst, void const *src, size_t bytes) const noexcept final;
    };

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_CPU_MEMORY_HH
