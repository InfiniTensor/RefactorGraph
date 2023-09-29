#ifndef COMPUTATION_LAYOUT_H
#define COMPUTATION_LAYOUT_H

namespace refactor::computation {

    enum class LayoutType {
        NCHW,
        NHWC,
        Others,
    };

}// namespace refactor::computation

#endif// COMPUTATION_LAYOUT_H
