#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <stdexcept>
#include <string>

inline std::string buildMsg(std::string &&msg, const char *file, int line) {
    msg += " Source ";
    msg += file;
    msg += ':';
    msg += std::to_string(line);
    return msg;
}

struct UnimplementError : public std::logic_error {
    explicit UnimplementError(std::string &&msg) : std::logic_error(msg) {}
};

#define RUNTIME_ERROR(msg) throw std::runtime_error(buildMsg(msg, __FILE__, __LINE__))
#define OUT_OF_RANGE(msg, a, b) throw std::out_of_range(buildMsg((std::to_string(a) + '/' + std::to_string(b) + ' ' + msg), __FILE__, __LINE__))
#define TODO(msg) throw UnimplementError(buildMsg(msg, __FILE__, __LINE__))

#ifndef DISABLE_ASSERT
#define ASSERT(condition, msg)                \
    {                                         \
        if (!(condition)) RUNTIME_ERROR(msg); \
    }
#else
#define ASSERT(condition, msg)
#endif

#endif// ERROR_HANDLER_H
