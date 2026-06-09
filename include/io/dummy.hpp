#ifndef KOCS_IO_DUMMY_HPP
#define KOCS_IO_DUMMY_HPP

namespace kocs::io {
  template<typename SimulationConfig>
  class Dummy {
    public:
      struct Settings { };

      Dummy(const std::string& path, const unsigned int agent_count, const Settings& settings) { }

      template<typename... Views>
      void write(unsigned int step, const Views&... views) { }

      unsigned int agent_count;

      inline void set_agent_count(const unsigned int value) {
        agent_count = value;
      }
  };
} // namespace kocs::io

#endif // KOCS_IO_DUMMY_HPP
