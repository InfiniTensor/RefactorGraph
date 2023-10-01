#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <fmt/format.h>
#include <stdexcept>

inline std::string buildMsg(std::string msg, const char *file, int line) {
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

#define RUNTIME_ERROR(msg) throw std::runtime_error(buildMsg(msg, __FILE__, __LINE__))
#define OUT_OF_RANGE(msg, a, b) throw std::out_of_range(buildMsg((std::to_string(a) + '/' + std::to_string(b) + ' ' + msg), __FILE__, __LINE__))
#define TODO(msg) throw UnimplementError(buildMsg(msg, __FILE__, __LINE__))

#define UNREACHABLEX(T, F, ...)                                                                               \
    [&]() -> T {                                                                                              \
        throw UnreachableError(buildMsg(fmt::format("Unreachable: " #F, ##__VA_ARGS__), __FILE__, __LINE__)); \
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
