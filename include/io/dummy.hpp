#ifndef KOCS_IO_DUMMY_HPP
#define KOCS_IO_DUMMY_HPP

namespace kocs::writers {
  template<typename SimulationConfig>
  class Dummy {
    public:
      Dummy(const std::string& path, const std::size_t flush_threshold = 65536) { }

      template<typename... Views>
      void write(unsigned int step, const Views&... views) { }
  };
} // namespace kocs::writers

#endif // KOCS_IO_DUMMY_HPP
