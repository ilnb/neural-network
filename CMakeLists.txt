cmake_minimum_required(VERSION 3.20)
project(neural-net LANGUAGES CXX)

set(NN_DTYPE "float" CACHE STRING "Type of NN::dtype")
set_property(CACHE NN_DTYPE PROPERTY STRINGS float double)
if(NOT NN_DTYPE MATCHES "^(float|double)$")
  message(FATAL_ERROR "NN_DTYPE must be 'float' or 'double', got ${NN_DTYPE}")
endif()

add_library(neural
  src/tensor/detail.cpp
  src/tensor/tensor.cpp
  src/nn/activations.cpp
  src/nn/loss.cpp
  src/autograd/functions.cpp
  src/optims.cpp
)

target_compile_features(neural PUBLIC cxx_std_23)

target_include_directories(neural
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(neural PUBLIC NN_DTYPE=${NN_DTYPE})

add_executable(tensor_test tests/tensor.cpp)
add_executable(act_test tests/act.cpp)
add_executable(loss_test tests/loss.cpp)
add_executable(autograd_test tests/autograd.cpp)
add_executable(demo_sgd demos/sgd.cpp)
add_executable(demo_xor demos/xor.cpp)

target_link_libraries(tensor_test PRIVATE neural)
target_link_libraries(act_test PRIVATE neural)
target_link_libraries(loss_test PRIVATE neural)
target_link_libraries(autograd_test PRIVATE neural)
target_link_libraries(demo_sgd PRIVATE neural)
target_link_libraries(demo_xor PRIVATE neural)
