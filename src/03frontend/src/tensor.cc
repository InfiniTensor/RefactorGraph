#include "frontend/tensor.h"
#include "common/error_handler.h"
#include <numeric>

namespace refactor::frontend {

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
                    RUNTIME_ERROR("Unreachable");
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

    Tensor::Tensor(common::DataType dataType_,
                   Shape shape_,
                   std::shared_ptr<common::Blob> data_,
                   std::unordered_set<DimVariable> depVariables_)
        : dataType(dataType_),
          shape(shape_),
          data(data_),
          depVariables(depVariables_) {}
    std::shared_ptr<Tensor>
    Tensor::share(common::DataType dt,
                  Shape shape,
                  std::unordered_set<DimVariable> depVariables,
                  std::shared_ptr<common::Blob> data) {
        return std::make_shared<Tensor>(dt, std::move(shape), std::move(data), std::move(depVariables));
    }
    bool Tensor::hasData() const { return data.get(); }
    int64_t Tensor::rank() const { return shape.size(); }
    size_t Tensor::elementsSize() const {
        return std::accumulate(shape.begin(), shape.end(), 1,
                               [](auto acc, auto const &it) { return acc * it.value(); });
    }
    size_t Tensor::bytesSize() const {
        return dataType.size() * elementsSize();
    }
    void *Tensor::malloc() {
        return (data = std::make_shared<common::Blob>(bytesSize()))->ptr;
    }
    void Tensor::free() {
        data = nullptr;
    }

}// namespace refactor::frontend
