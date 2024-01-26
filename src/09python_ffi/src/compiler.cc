#include "compiler.h"
#include "hardware/device_manager.h"
#include "kernel/allocators.h"
#include <execution>
#include <filesystem>
#include <fstream>

namespace refactor::python_ffi {
    using namespace frontend;
    namespace py = pybind11;

    Compiler::Compiler(frontend::Graph g) : _g(std::move(g)) {}

    void
    Compiler::substitute(CStr name, int64_t value) {
        if (!_g.substitute(name, value)) {
            fmt::println("\x1b[93mWARNING: variable \"{}\" not exist\x1b[0m", name);
        }
    }

    void
    Compiler::setInput(size_t index, pybind11::array data) {
        ASSERT(index < _g.internal().topology.globalInputsCount(),
               "Input {} not exist", index);

        Shape shape(data.ndim(), DimExpr(1));
        std::transform(std::execution::unseq,
                       data.shape(), data.shape() + data.ndim(), shape.begin(),
                       [](auto const &d) { return DimExpr(d); });
        auto ans = Tensor::share(parseNumpyDType(data.dtype()), std::move(shape), {});
        std::memcpy(ans->malloc(), data.data(), data.nbytes());
        _g.internal().edges[index].tensor = std::move(ans);
    }

    void
    Compiler::setInputInfo(size_t index, int dataType, DimVec dims) {
        ASSERT(index < _g.internal().topology.globalInputsCount(),
               "Input {} not exist", index);

        auto dataType_ = *DataType::parse(dataType);
        _g.internal().edges[index].tensor = Tensor::share(dataType_, dimVec2Shape(dims), {});
    }

    std::unordered_set<std::string>
    Compiler::fillEdgeInfo(bool calculate) { return _g.fillEdgeInfo(calculate); }

    Arc<Executor> Compiler::compileOn(
        Arc<hardware::Device> device,
        std::string allocator,
        std::vector<std::string> passes) {
        _g.collectVariables();
        std::vector<std::string_view> unknownVariables;
        for (auto const &[_, v] : _g.variables()) {
            if (!v->value.has_value()) {
                unknownVariables.emplace_back(v->name);
            }
        }
        if (!unknownVariables.empty()) {
            std::string msg = "Unknown variables: [ ";
            for (auto const &v : unknownVariables) {
                msg += v;
                msg += ' ';
            }
            msg += ']';
            RUNTIME_ERROR(std::move(msg));
        }

        std::unordered_set<std::string> passes_;
        passes_.reserve(passes.size());
        for (auto &p : passes) { passes_.emplace(std::move(p)); }

        _g.fillEdgeInfo(passes_.erase("ce"));

        auto computation = _g.lower();
        if (passes_.contains("lp")) {
            computation.layoutPermute();
        }

        auto kernel = computation.lower(device->type());
        auto stream = kernel.lower(std::move(device),
                                   allocator == "flat"
                                       ? kernel::flatAllocate
                                       : kernel::reusableAllocate);

        return std::make_shared<Executor>(
            std::move(computation),
            std::move(stream));
    }

    Arc<Executor>
    Compiler::compile(std::string target,
                      std::string allocator,
                      std::vector<std::string> passes) {
        using Target = hardware::Device::Type;
        // clang-format off
        auto target_ = target == "cpu"  ? Target::Cpu
                     : target == "cuda" ? Target::Nvidia
                     : UNREACHABLEX(Target, "Unknown target: {}", target);
        // clang-format on
        return compileOn(hardware::device::fetch(target_),
                         std::move(allocator),
                         std::move(passes));
    }

    std::vector<pybind11::array>
    Compiler::zeroInputs() const {
        std::vector<pybind11::array> ans;
        ans.reserve(_g.internal().topology.globalInputsCount());
        for (auto i : _g.internal().topology.globalInputs()) {
            auto const &tensor = *_g.internal().edges[i].tensor;
            ASSERT(!tensor.data, "Input tensor should not have data");

            std::vector<int64_t> shape(tensor.rank());
            std::transform(std::execution::unseq,
                           tensor.shape.begin(), tensor.shape.end(), shape.begin(),
                           [](auto const &d) { return d.value(); });
            ans.push_back(py::array(buildNumpyDType(tensor.dataType), std::move(shape)));
        }
        return ans;
    }

    std::optional<py::array>
    Compiler::getTensor(CStr name) const {
        auto const &edges = _g.internal().edges;
        auto it = std::find_if(edges.begin(), edges.end(),
                               [name](auto const &edge) { return edge.name == name; });
        if (it == edges.end()) { return std::nullopt; }
        auto const &tensor = *it->tensor;

        std::vector<int64_t> shape(tensor.shape.size());
        std::transform(std::execution::unseq,
                       tensor.shape.begin(), tensor.shape.end(), shape.begin(),
                       [](auto const &d) { return d.value(); });

        auto ans = py::array(buildNumpyDType(tensor.dataType), std::move(shape));
        if (tensor.data) { std::memcpy(ans.mutable_data(), tensor.data->get<void>(), ans.nbytes()); }
        return ans;
    }

    void
    Compiler::serialize(std::string path_) {
        _g.collectVariables();
        std::vector<std::string_view> unknownVariables;
        for (auto const &[_, v] : _g.variables()) {
            if (!v->value.has_value()) {
                unknownVariables.emplace_back(v->name);
            }
        }
        if (!unknownVariables.empty()) {
            std::string msg = "Unknown variables: [ ";
            for (auto const &v : unknownVariables) {
                msg += v;
                msg += ' ';
            }
            msg += ']';
            RUNTIME_ERROR(std::move(msg));
        }
        _g.fillEdgeInfo(false);

        namespace fs = std::filesystem;
        auto path = fs::path(std::move(path_));
        fs::create_directories(path);
        ASSERT(fs::is_directory(path), "Failed to create \"{}\"", path.c_str());

        auto [text, data] = _g.lower().serialize(true);
        std::ofstream(path / "graph.info") << std::move(text);
        std::ofstream(path / "graph.data", std::ios::binary)
            .write(reinterpret_cast<const char *>(data.data()), data.size());
    }

}// namespace refactor::python_ffi
