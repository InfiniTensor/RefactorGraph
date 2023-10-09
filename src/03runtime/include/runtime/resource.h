#ifndef RUNTIME_RESOURCES_H
#define RUNTIME_RESOURCES_H

#include <any>
#include <memory>
#include <string_view>
#include <unordered_map>

namespace refactor::runtime {

    class Resource {
    public:
        virtual size_t resourceTypeId() const = 0;
        virtual std::string_view description() const = 0;
        bool is(size_t) const noexcept;
    };

    using ResourceBox = std::unique_ptr<Resource>;

    class Resources {
        std::unordered_map<size_t, std::unique_ptr<Resource>> _internal;

    public:
        Resource *fetch(size_t) noexcept;
        Resource *fetchOrStore(ResourceBox) noexcept;
        Resource *fetchOrStore(size_t, ResourceBox()) noexcept;
    };

}// namespace refactor::runtime

#endif// RUNTIME_RESOURCES_H
