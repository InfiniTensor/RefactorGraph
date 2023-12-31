﻿#ifndef RUNTIME_RESOURCES_H
#define RUNTIME_RESOURCES_H

#include <functional>
#include <memory>
#include <string_view>

namespace refactor::runtime {

    class Resource {
    public:
        virtual ~Resource() = default;
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
        Resource *fetchOrStore(size_t, std::function<ResourceBox()>);

        template<class T> T *fetch() noexcept {
            return dynamic_cast<T *>(fetch(T::typeId()));
        }
        template<class T> T *fetch(size_t id) noexcept {
            return dynamic_cast<T *>(fetch(id));
        }
        template<class T> T *fetchOrStore(ResourceBox r) noexcept {
            return dynamic_cast<T *>(fetchOrStore(std::move(r)));
        }
        template<class T> T *fetchOrStore(size_t id, ResourceBox f()) {
            return dynamic_cast<T *>(fetchOrStore(id, f));
        }
        template<class T, class... Args> T *fetchOrStore(Args &&...args) {
            return dynamic_cast<T *>(fetchOrStore(
                T::typeId(),
                [&]() -> ResourceBox { return T::build(std::forward<Args>(args)...); }));
        }
    };

}// namespace refactor::runtime

#endif// RUNTIME_RESOURCES_H
