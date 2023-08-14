#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <stdexcept>
#include <string>

inline std::string buildMsg(const char *msg, const char *file, int line) {
    std::string ans = msg;
    ans += " Source ";
    ans += file;
    ans += ':';
    ans += std::to_string(line);
    return ans;
}

struct UnimplementError : public std::logic_error {
    explicit UnimplementError(const char *msg) : std::logic_error(msg) {}
};

#define RUNTIME_ERROR(msg) throw std::runtime_error(buildMsg(msg, __FILE__, __LINE__))
#define OUT_OF_RANGE(msg, a, b) throw std::out_of_range( \
    buildMsg((std::to_string(a) + '/' + std::to_string(b) + ' ' + msg).c_str(), __FILE__, __LINE__))
#define TODO(msg) throw UnimplementError(buildMsg(msg, __FILE__, __LINE__).c_str())

#endif// ERROR_HANDLER_H
