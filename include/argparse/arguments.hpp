#ifndef KOCS_ARGPARSE_ARGUMENT_PARSER_HPP
#define KOCS_ARGPARSE_ARGUMENT_PARSER_HPP

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <iostream>

// https://github.com/p-ranav/argparse
#include <argparse/argparse.hpp>

namespace kocs {
  template<typename T>
  struct shape_for {
    static_assert(!sizeof(T), "No default scan char for this type");
  };

  template<>
  struct shape_for<float> {
    static constexpr char value = 'g';
  };

  template<>
  struct shape_for<double> {
    static constexpr char value = 'g';
  };

  template<>
  struct shape_for<char> {
    static constexpr char value = 'i';
  };

  template<>
  struct shape_for<long long> {
    static constexpr char value = 'i';
  };

  template<>
  struct shape_for<unsigned long long> {
    static constexpr char value = 'u';
  };

  template<>
  struct shape_for<std::string> {
    static constexpr char value = '_';
  };

  template<>
  struct shape_for<int8_t> {
    static constexpr char value = 'i';
  };

  template<>
  struct shape_for<uint8_t> {
    static constexpr char value = 'u';
  };

  template<>
  struct shape_for<int16_t> {
    static constexpr char value = 'i';
  };

  template<>
  struct shape_for<uint16_t> {
    static constexpr char value = 'u';
  };

  template<>
  struct shape_for<int32_t> {
    static constexpr char value = 'i';
  };

  template<>
  struct shape_for<uint32_t> {
    static constexpr char value = 'u';
  };

  template<>
  struct shape_for<int64_t> {
    static constexpr char value = 'i';
  };

  template<>
  struct shape_for<uint64_t> {
    static constexpr char value = 'u';
  };

  struct Arguments {
    argparse::ArgumentParser parser;

    Arguments(const std::string& name) : parser(name) { }

    template<char Shape, typename T, typename U = T, typename... V>
    Arguments& add_argument(
      const std::string& short_name,
      const std::string& long_name,
      T& storage,
      const U default_value = U{},
      const std::string& help = "",
      V&&... choices
    ) {
      auto argument = parser.add_argument(short_name, long_name)
        .template scan<shape_for<T>::value, T>()
        .default_value(static_cast<T>(default_value))
        .store_into(storage)
        .help(help);
      if constexpr (sizeof...(choices) > 0)
        argument.choices(std::forward<V>(choices)...);

      return *this;
    }

    template<typename T, typename U = T, typename... V>
    Arguments& add_argument(
      const std::string& short_name,
      const std::string& long_name,
      T& storage,
      const U default_value = U{},
      const std::string& help = "",
      V&&... choices
    ) {
      if constexpr (std::is_same_v<T, std::string> == false) {
        return this->template add_argument<shape_for<T>::value, T, U>(
          short_name, long_name, storage, default_value, help, std::forward<V>(choices)...
        );
      }

      auto argument = parser.add_argument(short_name, long_name)
        .default_value(static_cast<T>(default_value))
        .store_into(storage)
        .help(help);
      if constexpr (sizeof...(choices) > 0)
        argument.choices(std::forward<V>(choices)...);
      
      return *this;
    }
    


    template<char Shape, typename T, typename... V>
    Arguments& add_required_argument(
      const std::string& short_name,
      const std::string& long_name,
      T& storage,
      const std::string& help = "",
      V&&... choices
    ) {
      auto argument = parser.add_argument(short_name, long_name)
        .template scan<shape_for<T>::value, T>()
        .required()
        .store_into(storage)
        .help(help);
      if constexpr (sizeof...(choices) > 0)
        argument.choices(std::forward<V>(choices)...);

      return *this;
    }

    template<typename T, typename... V>
    Arguments& add_required_argument(
      const std::string& short_name,
      const std::string& long_name,
      T& storage,
      const std::string& help = "",
      V&&... choices
    ) {
      if constexpr (std::is_same_v<T, std::string> == false) {
        return this->template add_argument<shape_for<T>::value, T>(
          short_name, long_name, storage, help, std::forward<V>(choices)...
        );
      }

      auto argument = parser.add_argument(short_name, long_name)
        .required()
        .store_into(storage)
        .help(help);
      if constexpr (sizeof...(choices) > 0)
        argument.choices(std::forward<V>(choices)...);
      
      return *this;
    }



    Arguments& add_flag(
      const std::string& short_name,
      const std::string& long_name,
      bool& storage,
      const std::string& help = ""
    ) {
      parser.add_argument(short_name, long_name)
        .help(help)
        .default_value(false)
        .implicit_value(true)
        .store_into(storage);

      return *this;
    }



    bool parse(int argc, char* argv[]) {
      try {
        parser.parse_args(argc, argv);
      }
      catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        return false;
      }
      return true;
    }
  };
}

#endif // KOCS_ARGPARSE_ARGUMENT_PARSER_HPP
