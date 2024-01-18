#include "computation/operators/simple_unary.h"

namespace refactor::computation {
    using Op = SimpleUnary;

    size_t Op::typeId(SimpleUnaryType type) noexcept {
        switch (type) {
            case SimpleUnaryType::Abs: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Acos: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Acosh: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Asin: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Asinh: {
                static uint8_t ID = 5;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Atan: {
                static uint8_t ID = 6;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Atanh: {
                static uint8_t ID = 7;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Cos: {
                static uint8_t ID = 8;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Cosh: {
                static uint8_t ID = 9;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Sin: {
                static uint8_t ID = 10;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Sinh: {
                static uint8_t ID = 11;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Tan: {
                static uint8_t ID = 12;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Tanh: {
                static uint8_t ID = 13;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Relu: {
                static uint8_t ID = 14;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Sqrt: {
                static uint8_t ID = 15;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Sigmoid: {
                static uint8_t ID = 16;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Erf: {
                static uint8_t ID = 17;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Neg: {
                static uint8_t ID = 18;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::Not: {
                static uint8_t ID = 19;
                return reinterpret_cast<size_t>(&ID);
            }
            case SimpleUnaryType::HardSwish: {
                static uint8_t ID = 20;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }
    size_t Op::opTypeId() const noexcept {
        return typeId(type);
    }
    std::string_view Op::name() const noexcept {
        switch (type) {
            case SimpleUnaryType::Abs:
                return "Abs";
            case SimpleUnaryType::Acos:
                return "Acos";
            case SimpleUnaryType::Acosh:
                return "Acosh";
            case SimpleUnaryType::Asin:
                return "Asin";
            case SimpleUnaryType::Asinh:
                return "Asinh";
            case SimpleUnaryType::Atan:
                return "Atan";
            case SimpleUnaryType::Atanh:
                return "Atanh";
            case SimpleUnaryType::Cos:
                return "Cos";
            case SimpleUnaryType::Cosh:
                return "Cosh";
            case SimpleUnaryType::Sin:
                return "Sin";
            case SimpleUnaryType::Sinh:
                return "Sinh";
            case SimpleUnaryType::Tan:
                return "Tan";
            case SimpleUnaryType::Tanh:
                return "Tanh";
            case SimpleUnaryType::Relu:
                return "Relu";
            case SimpleUnaryType::Sqrt:
                return "Sqrt";
            case SimpleUnaryType::Sigmoid:
                return "Sigmoid";
            case SimpleUnaryType::Erf:
                return "Erf";
            case SimpleUnaryType::Neg:
                return "Neg";
            case SimpleUnaryType::Not:
                return "Not";
            case SimpleUnaryType::HardSwish:
                return "HardSwish";
            default:
                UNREACHABLE();
        }
    }

    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        return std::make_unique<kernel::SimpleUnaryCollector>(target, type);
    }
    auto Op::serialize() const noexcept -> std::string {
        return fmt::format("{}()", name());
    }

}// namespace refactor::computation
