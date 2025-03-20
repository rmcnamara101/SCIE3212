cmake_minimum_required(VERSION 3.10)
project(cpp_simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP if available
find_package(OpenMP)

# Enable optimization flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native)
endif()

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(cpp_simulation MODULE
    simulation.cpp
    bindings.cpp
)

target_link_libraries(cpp_simulation PRIVATE
    pybind11::module
    Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cpp_simulation PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(cpp_simulation PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

if(APPLE)
    set_target_properties(cpp_simulation PROPERTIES
        SUFFIX ".so"
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif()

# Enable link time optimization
set_target_properties(cpp_simulation PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
