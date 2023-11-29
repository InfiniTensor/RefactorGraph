#ifndef KERNEL_CODE_REPO_H
#define KERNEL_CODE_REPO_H

#include <filesystem>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace refactor::kernel {

    class CodeRepo {
    public:
        virtual ~CodeRepo() = default;
        void *compile(
            const char *dir,
            const char *code,
            const char *symbol);
        void *fetch(const char *dir);

    protected:
        std::unordered_map<std::string, void *> _dirs;

        virtual std::string_view hardware() const = 0;
        virtual std::string_view extension() const = 0;
        virtual void *_compile(std::filesystem::path const &src,
                               const char *symbol) = 0;

        static std::filesystem::path const &repo_path();
    };

}// namespace refactor::kernel

#endif// KERNEL_CODE_REPO_H
