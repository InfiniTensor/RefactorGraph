#include "cuda_code_repo.hh"
#include "common.h"
#include <dlfcn.h>

namespace refactor::kernel {

    auto CudaCodeRepo::hardware() const noexcept -> std::string_view {
        return "CUDA";
    }
    auto CudaCodeRepo::extension() const noexcept -> std::string_view {
        return "cu";
    }
    void *CudaCodeRepo::_compile(std::filesystem::path const &src,
                                 const char *symbol) {
        auto out = src, so = src;
        out.replace_extension("o");
        so.replace_filename("libkernel.so");
        {
            std::string command;
            command = fmt::format("nvcc -Xcompiler \"-fPIC\" {} -c -o {}", src.c_str(), out.c_str());
            std::system(command.c_str());
            command = fmt::format("nvcc -shared {} -o {}", out.c_str(), so.c_str());
            std::system(command.c_str());
        }

        auto handle = dlopen(so.c_str(), RTLD_LAZY);
        ASSERT(handle, "Failed to load kernel library: {}", dlerror());
        auto function = dlsym(handle, symbol);
        ASSERT(function, "Failed to load kernel function: {}", dlerror());
        return function;
    }
    void *CudaCodeRepo::compile_(
        const char *dir,
        const char *code,
        const char *symbol) {
        static CudaCodeRepo repo;
        return repo.compile(dir, code, symbol);
    }
    std::string_view CudaCodeRepo::memCopyType(size_t size) {
        return size == 1    ? "char"
               : size == 2  ? "short"
               : size == 4  ? "float"
               : size == 8  ? "float2"
               : size == 16 ? "float4"
               : size == 32 ? "double4"
                            : UNREACHABLEX(const char *, "");
    }

}// namespace refactor::kernel
