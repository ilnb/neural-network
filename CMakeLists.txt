cmake_minimum_required(VERSION 3.20)
project(neural-net LANGUAGES CXX)

add_library(neural
  src/tensor/tensor.cpp
  src/nn/activations.cpp
  src/nn/loss.cpp
  src/autograd/functions.cpp
)

target_compile_features(neural PUBLIC cxx_std_23)

target_include_directories(neural
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(tensor_test tests/tensor.cpp)
add_executable(act_test tests/act.cpp)
add_executable(loss_test tests/loss.cpp)
target_link_libraries(tensor_test PRIVATE neural)
target_link_libraries(act_test PRIVATE neural)
target_link_libraries(loss_test PRIVATE neural)
