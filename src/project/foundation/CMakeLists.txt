find_package(pylon 7.4.0 REQUIRED)

pybind11_add_module(foundation python.cpp example.cpp)
target_link_libraries(foundation PUBLIC spdlog::spdlog Tracy::TracyClient)
target_include_directories(foundation PUBLIC ../../include)
