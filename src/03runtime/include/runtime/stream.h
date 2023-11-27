#ifndef RUNTIME_STREAM_H
#define RUNTIME_STREAM_H

#include "graph_topo.h"
#include "hardware/device.h"
#include "resource.h"
#include <chrono>
#include <functional>
#include <variant>

namespace refactor::runtime {
    using Routine = std::function<void(runtime::Resources &, void *, void const *const *, void *const *)>;

    void emptyRoutine(runtime::Resources &, void *, void const *const *, void *const *);

    struct Address {
        std::variant<size_t, Arc<hardware::Device::Blob>> value;

        void *operator()(void *stack) const;

        bool isBlob() const noexcept;
        bool isOffset() const noexcept;

        auto blob() const noexcept -> hardware::Device::Blob const &;
        auto offset() const noexcept -> size_t;
    };

    struct Node {
        Routine routine;
        size_t workspaceOffset;

        template<class T>
        Node(T &&r, size_t wso = 0) noexcept
            : routine(std::forward<T>(r)),
              workspaceOffset(wso) {}
    };

    class Stream {
        using _N = Node;
        using _E = Address;
        using _G = graph_topo::Graph<_N, _E>;

        Resources _resources;
        Arc<hardware::Device> _device;
        size_t _stackSize;
        std::vector<size_t> _outputsSize;
        _G _internal;

        Arc<hardware::Device::Blob> _stack;

    public:
        Stream(decltype(_resources),
               decltype(_device),
               decltype(_stackSize),
               decltype(_outputsSize),
               graph_topo::GraphTopo,
               std::vector<_N>,
               std::vector<_E>);
        void setData(count_t, void const *, size_t);
        void setData(count_t, Arc<hardware::Device::Blob>);
        void getData(count_t, void *, size_t) const;
        auto prepare() -> std::vector<count_t>;
        void run();
        auto bench(void (*sync)()) -> std::vector<std::chrono::nanoseconds>;
        void trace(std::function<void(count_t, void const *const *, void const *const *)>);
    };

}// namespace refactor::runtime

#endif// RUNTIME_STREAM_H
