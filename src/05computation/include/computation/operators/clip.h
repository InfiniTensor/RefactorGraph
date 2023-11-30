#ifndef COMPUTATION_CLIP_H
#define COMPUTATION_CLIP_H

#include "../operator.h"

namespace refactor::computation {

    struct Clip final : public Operator {

        Clip() = default;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CLIP_H
