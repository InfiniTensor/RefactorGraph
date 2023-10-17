// #include "computation/operators/einsum.h"
#include "einsum.hh"
#include "common.h"
#include "refactor/common.h"
#include <variant>

namespace refactor::onnx {
    using Op = Einsum;

    Op::Einsum(std::string equation_)
        : Operator(), equation(std::move(equation_)) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        return OpBox(std::make_unique<Op>(std::move(attributes.at("equation").string())));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Einsum"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }

        std::vector<uint_lv2> indices;
        std::vector<char> data;
        {
            auto implicit = true;
            for (size_t i = 0; i < equation.size();) {
                switch (equation[i]) {
                    case ' ':
                        ++i;
                        break;

                    case ',':
                        ++i;
                        ASSERT(implicit, "");
                        indices.push_back(data.size());
                        break;

                    case '-':
                        ASSERT(i + 1 < equation.size(), "");
                        ASSERT(equation[i + 1] == '>', "");
                        i += 2;
                        implicit = false;
                        indices.push_back(data.size());
                        break;

                    case '.':
                        ASSERT(i + 2 < equation.size(), "");
                        ASSERT(equation[i + 1] == '.' && equation[i + 2] == '.', "");
                        i += 3;
                        data.push_back('.');
                        break;

                    default:
                        if ('a' <= equation[i] && equation[i] <= 'z') {
                            data.push_back(equation[i++]);
                        } else {
                            UNREACHABLE();
                        }
                        break;
                }
            }
            if (implicit) {
                indices.push_back(data.size());
            }
        }
        // ---
        {
            auto implicit = indices.back() == data.size();
            std::string msg;
            if (!implicit) {
                for (auto i : range(indices.rbegin()[0], static_cast<uint_lv2>(data.size()))) {
                    msg += data[i];
                }
                msg += " = ";
            }
            uint_lv2 begin = 0;
            for (auto i : range0_(indices.size())) {
                for (auto j : range(begin, indices[i])) {
                    msg += data[j];
                }
                msg += " * ";
                begin = indices[i];
            }
            fmt::println("{}", msg.substr(0, msg.size() - 3));
        }
        TODO("");
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        TODO("");
    }

}// namespace refactor::onnx
