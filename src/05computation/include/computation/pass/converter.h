#ifndef COMPUTATION_CONVERTER_H
#define COMPUTATION_CONVERTER_H

#include "../graph.h"

namespace refactor::computation {

    class Converter {
    public:
        Converter() = default;
        virtual ~Converter() = default;
        virtual bool execute(const std::shared_ptr<GraphMutant> &) const = 0;
        static Converter *get(std::string_view key) {
            //fmt::println("{}", storage().size());
            if (storage().find(key) != storage().end()) {
                return storage().at(key).get();
            }
            return nullptr;
        };
        static void add(std::shared_ptr<Converter> converter, std::string_view key) {
            storage().insert(std::make_pair(key, converter));
        };
        static std::unordered_map<std::string_view, std::shared_ptr<Converter>> &storage() {
            static std::unordered_map<std::string_view, std::shared_ptr<Converter>> passStorage;
            return passStorage;
        }
    };

    template<class T>
    class ConverterRegister {
    public:
        ConverterRegister(const char *claim) {
            T *instance = new T;
            Converter::add(std::shared_ptr<Converter>(instance), claim);
        }
    };

}// namespace refactor::computation

#endif// COMPUTATION_CONVERTER_H