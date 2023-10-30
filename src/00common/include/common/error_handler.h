#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <fmt/format.h>
#include <stdexcept>

namespace refactor::error {
    inline std::string buildMsg(std::string msg, const char *file, int line) noexcept {
        msg += " Source ";
        msg += file;
        msg += ':';
        msg += std::to_string(line);
        return msg;
    }

    struct UnimplementError : public std::logic_error {
        explicit UnimplementError(std::string msg)
            : std::logic_error(std::move(msg)) {}
    };

    struct UnreachableError : public std::logic_error {
        explicit UnreachableError(std::string msg)
            : std::logic_error(std::move(msg)) {}
    };

}// namespace refactor::error

#define ERROR_MSG(msg) refactor::error::buildMsg(msg, __FILE__, __LINE__)
#define RUNTIME_ERROR(msg) throw std::runtime_error(ERROR_MSG(msg))
#define OUT_OF_RANGE(msg, a, b) throw std::out_of_range(ERROR_MSG((std::to_string(a) + '/' + std::to_string(b) + ' ' + msg)))
#define TODO(msg) throw refactor::error::UnimplementError(ERROR_MSG(msg))

#define UNREACHABLEX(T, F, ...)                                                                             \
    [&]() -> T {                                                                                            \
        throw refactor::error::UnreachableError(ERROR_MSG(fmt::format("Unreachable: " #F, ##__VA_ARGS__))); \
    }()
#define UNREACHABLE() UNREACHABLEX(void, "no message")

#ifndef DISABLE_ASSERT
#define ASSERT(condition, msg)                \
    {                                         \
        if (!(condition)) RUNTIME_ERROR(msg); \
    }
#else
#define ASSERT(condition, msg)
#endif

#endif// ERROR_HANDLER_H
