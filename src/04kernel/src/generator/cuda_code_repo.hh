#ifndef KRENEL_CUDA_CODE_REPO_HH
#define KRENEL_CUDA_CODE_REPO_HH

#include "kernel/generator/code_repo.h"

namespace refactor::kernel {

    class CudaCodeRepo final : public CodeRepo {
    public:
        static void *compile_(
            const char *dir,
            const char *code,
            const char *symbol);
        static std::string_view memCopyType(size_t);

    protected:
        std::string_view hardware() const noexcept final;
        std::string_view extension() const noexcept final;
        void *_compile(std::filesystem::path const &src,
                       const char *symbol) final;
    };

}// namespace refactor::kernel

#endif// KRENEL_CUDA_CODE_REPO_HH
