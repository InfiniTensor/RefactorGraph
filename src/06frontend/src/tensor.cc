#include "frontend/tensor.h"
#include "refactor/common.h"
#include <execution>
#include <numeric>

namespace refactor::frontend {
    using namespace mem_manager;

    DimVariableInternal::DimVariableInternal(std::string name_, std::optional<int64_t> value_)
        : name(std::move(name_)), value(value_) {}

    DimExpr::DimExpr(int64_t val) : expr(val) {}
    DimExpr::DimExpr(std::string name)
        : expr(std::make_shared<DimVariableInternal>(std::move(name))) {}
    DimExpr::DimExpr(DimVariable var)
        : expr(std::move(var)) {}

    bool DimExpr::operator==(DimExpr const &rhs) const {
        if (expr.index() != rhs.expr.index()) {
            return false;
        } else {
            switch (expr.index()) {
                case 0:
                    return std::get<0>(expr) == std::get<0>(rhs.expr);
                case 1:
                    return std::get<1>(expr) == std::get<1>(rhs.expr);
                default:
                    UNREACHABLE();
            }
        }
    }
    bool DimExpr::operator!=(DimExpr const &rhs) const { return !operator==(rhs); }
    bool DimExpr::isValue() const { return std::holds_alternative<int64_t>(expr); }
    bool DimExpr::isVariable() const { return std::holds_alternative<DimVariable>(expr); }
    bool DimExpr::hasValue() const { return isValue() || variable()->value; }
    int64_t DimExpr::value() const { return isValue() ? std::get<int64_t>(expr) : variable()->value.value(); }
    DimVariable DimExpr::variable() const { return std::get<DimVariable>(expr); }

    std::string shapeFormat(Shape const &shape) {
        std::string ans("Shape{ ");
        for (auto const &it : shape) {
            if (it.isValue()) {
                ans += std::to_string(it.value());
            } else {
                auto const &var = it.variable();
                ans += var->name;
                if (var->value) {
                    ans += ":";
                    ans += std::to_string(*var->value);
                }
            }
            ans += " ";
        }
        ans += "}";
        return ans;
    }

    bool TensorSnapshot::operator==(TensorSnapshot const &rhs) const {
        return dataType == rhs.dataType &&
               shape == rhs.shape &&
               dataPtr.lock().get() == rhs.dataPtr.lock().get();
    }
    bool TensorSnapshot::operator!=(TensorSnapshot const &rhs) const { return !operator==(rhs); }
    bool TensorSnapshot::operator==(Tensor const &rhs) const {
        if (dataType != rhs.dataType ||
            shape.size() != rhs.shape.size() ||
            dataPtr.lock().get() != rhs.data.get()) {
            return false;
        }
        for (auto i : range0_(shape.size())) {
            auto hasValue = std::holds_alternative<int64_t>(shape[i]);
            if (hasValue != rhs.shape[i].hasValue()) {
                return false;
            }
            if (hasValue) {
                if (std::get<int64_t>(shape[i]) != rhs.shape[i].value()) {
                    return false;
                }
            } else {
                if (std::get<DimVariable>(shape[i]) != rhs.shape[i].variable()) {
                    return false;
                }
            }
        }
        return true;
    }
    bool TensorSnapshot::operator!=(Tensor const &rhs) const { return !operator==(rhs); }

    Tensor::Tensor(DataType dataType_, Shape shape_, std::shared_ptr<Blob> data_, std::unordered_set<DimVariable> depVariables_)
        : dataType(dataType_),
          shape(std::move(shape_)),
          data(std::move(data_)),
          depVariables(std::move(depVariables_)) {}
    std::shared_ptr<Tensor>
    Tensor::share(const Tensor &rhs) {
        return std::make_shared<Tensor>(rhs);
    }
    std::shared_ptr<Tensor>
    Tensor::share(DataType dt,
                  Shape shape,
                  std::unordered_set<DimVariable> depVariables,
                  std::shared_ptr<Blob> data) {
        return std::make_shared<Tensor>(dt, std::move(shape), std::move(data), std::move(depVariables));
    }
    bool Tensor::hasData() const { return data.get(); }
    int64_t Tensor::rank() const { return shape.size(); }
    size_t Tensor::elementsSize() const {
        return std::accumulate(shape.begin(), shape.end(), 1,
                               [](auto acc, auto const &it) { return acc * it.value(); });
    }
    size_t Tensor::bytesSize() const { return dataType.size() * elementsSize(); }
    TensorSnapshot Tensor::snapshot() const {
        ShapeSnapshot shape_(shape.size());
        std::transform(std::execution::unseq,
                       shape.begin(), shape.end(), shape_.begin(),
                       [](auto const &it) -> std::variant<int64_t, DimVariable> {
                           if (it.hasValue()) {
                               return it.value();
                           } else {
                               return it.variable();
                           }
                       });
        return TensorSnapshot{dataType, std::move(shape_), data};
    }
    void *Tensor::malloc() {
        auto [data_, ptr] = Blob::share(bytesSize());
        data = std::move(data_);
        return ptr;
    }
    void Tensor::free() {
        data = nullptr;
    }

    TensorRefs::TensorRefs(
        std::vector<Edge> const &edges,
        slice_t<graph_topo::idx_t> slice)
        : _edges(edges), _slice(slice) {}
    Tensor const &TensorRefs::operator[](size_t i) const {
        return *_edges[_slice[i]].tensor;
    }
    size_t TensorRefs::size() const {
        return _slice.size();
    }
    bool TensorRefs::empty() const {
        return _slice.empty();
    }

    TensorRefs::Iterator::Iterator(TensorRefs const &internal, size_t i)
        : _internal(internal), _index(i) {}

    bool TensorRefs::Iterator::operator==(Iterator const &rhs) const {
        return _index == rhs._index;
    }
    bool TensorRefs::Iterator::operator!=(Iterator const &rhs) const {
        return !operator==(rhs);
    }
    Tensor const &TensorRefs::Iterator::operator*() const {
        return _internal[_index];
    }
    TensorRefs::Iterator &TensorRefs::Iterator::operator++() {
        ++_index;
        return *this;
    }

    auto TensorRefs::begin() const -> Iterator {
        return Iterator(*this, 0);
    }
    auto TensorRefs::end() const -> Iterator {
        return Iterator(*this, size());
    }

}// namespace refactor::frontend
