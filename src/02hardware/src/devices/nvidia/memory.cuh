#ifndef HARDWARE_DEVICES_NVIDIA_MEMORY_CUH
#define HARDWARE_DEVICES_NVIDIA_MEMORY_CUH

#include "hardware/memory.h"

namespace refactor::hardware {

    class NvidiaMemory final : public Memory {
        void *malloc(size_t) final;
        void free(void *) final;
        void *copyHD(void *dst, void const *src, size_t bytes) const final;
        void *copyDH(void *dst, void const *src, size_t bytes) const final;
        void *copyDD(void *dst, void const *src, size_t bytes) const final;
    };

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_NVIDIA_MEMORY_CUH
