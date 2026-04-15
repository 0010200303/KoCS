cmake_minimum_required(VERSION 3.20)

project(KoCS
  VERSION 0.1
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Kokkos_ENABLE_DEPRECATED_CODE OFF CACHE BOOL "" FORCE)
set(Kokkos_BACKEND "" CACHE STRING
  "Choose Kokkos backend: SERIAL, OPENMP, THREADS, HPX, CUDA, HIP, SYCL, OPENMPTARGET, OPENACC"
)

# TODO: error if backend is unknown
if (Kokkos_BACKEND STREQUAL "OPENMP")
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "THREADS")
  set(Kokkos_ENABLE_THREADS ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "HPX")
  set(Kokkos_ENABLE_HPX ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "CUDA")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "HIP")
  set(Kokkos_ENABLE_HIP ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "SYCL")
  set(Kokkos_ENABLE_SYCL ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "OPENMPTARGET")
  set(Kokkos_ENABLE_OPENMPTARGET ON CACHE BOOL "" FORCE)
elseif (Kokkos_BACKEND STREQUAL "OPENACC")
  set(Kokkos_ENABLE_OPENACC ON CACHE BOOL "" FORCE)
else()
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
endif()

find_package(Kokkos 5.0 CONFIG)
if (Kokkos_FOUND)
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version ${Kokkos_VERSION})")
else()
  message(STATUS "Kokkos not found externally. Fetching via FetchContent.")

  include(FetchContent)

  FetchContent_Declare(
    Kokkos
    URL https://github.com/kokkos/kokkos/releases/download/5.0.0/kokkos-5.0.0.tar.gz
    URL_HASH SHA256=c45f3e19c3eb71fc8b7210cb04cac658015fc1839e7cc0571f7406588ff9bcef
  )
  FetchContent_MakeAvailable(Kokkos)
endif()

find_package(HighFive CONFIG QUIET)
if (HighFive_FOUND)
    message(STATUS "Found HighFive: ${HighFive_DIR}")
else()
    message(STATUS "HighFive not found externally. Fetching...")

    include(FetchContent)

    FetchContent_Declare(
        HighFive
        GIT_REPOSITORY https://github.com/highfive-devs/highfive
        GIT_TAG v3.3.0
        # TODO: URL_HASH
    )
    FetchContent_MakeAvailable(HighFive)
endif()

find_package(HDF5 REQUIRED COMPONENTS C CXX)

add_library(KoCS_core INTERFACE)

target_include_directories(KoCS_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(KoCS_core INTERFACE
  Kokkos::kokkos
  HighFive::HighFive
  HDF5::HDF5
)

target_sources(KoCS_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/runtime_guard.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/utils.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/vector.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/initializers/line.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/initializers/random_spheres.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/simulation_config.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/simulation.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/include/io/writer.hpp
)



function(add_KoCS_simulation target_name source_file)
  add_executable(${target_name} ${source_file})

  target_link_libraries(${target_name} PRIVATE
    KoCS_core
  )

  target_compile_options(${target_name} PRIVATE -O3)
endfunction()

set(KOCS_USER_TARGET "kocs"
  CACHE STRING "Name of the generated user simulation target"
)

set(KOCS_USER_MAIN ""
  CACHE FILEPATH "Path to a user-defined simulation source file"
)

if (KOCS_USER_MAIN)
  message(STATUS
    "Configuring user simulation from ${KOCS_USER_MAIN} "
    "as target ${KOCS_USER_TARGET}"
  )
  add_KoCS_simulation(${KOCS_USER_TARGET} "${KOCS_USER_MAIN}")
endif()
