cmake_minimum_required(VERSION 3.20)
project(neural-net LANGUAGES CXX)

set(NN_DTYPE "float" CACHE STRING "Type of NN::dtype")
set_property(CACHE NN_DTYPE PROPERTY STRINGS float double)
if(NOT NN_DTYPE MATCHES "^(float|double)$")
  message(FATAL_ERROR "NN_DTYPE must be 'float' or 'double', got ${NN_DTYPE}")
endif()

add_library(neural SHARED
  src/tensor/detail.cpp
  src/tensor/tensor.cpp
  src/activations/activations.cpp
  src/regularization/regularization.cpp
  src/loss/loss.cpp
  src/autograd/functions.cpp
  src/optims/optims.cpp
  src/data/data.cpp
  src/data/split.cpp
  src/module/batchnorm.cpp
  src/module/conv.cpp
)

add_compile_options(
  -Wall
  -Wextra
  -Wunused
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
add_executable(reg_test tests/regularization.cpp)
add_executable(data_test tests/data.cpp)
add_executable(bn_test tests/batchnorm.cpp)
add_executable(conv_test tests/conv.cpp)
add_executable(demo_sgd demos/sgd.cpp)
add_executable(demo_xor demos/xor.cpp)
add_executable(demo_digits_mlp demos/digits_mlp.cpp)
add_executable(demo_digits_cnn demos/digits_cnn.cpp)

target_link_libraries(tensor_test PRIVATE neural)
target_link_libraries(act_test PRIVATE neural)
target_link_libraries(loss_test PRIVATE neural)
target_link_libraries(autograd_test PRIVATE neural)
target_link_libraries(reg_test PRIVATE neural)
target_link_libraries(data_test PRIVATE neural)
target_link_libraries(bn_test PRIVATE neural)
target_link_libraries(conv_test PRIVATE neural)
target_link_libraries(demo_sgd PRIVATE neural)
target_link_libraries(demo_xor PRIVATE neural)
target_link_libraries(demo_digits_mlp PRIVATE neural)
target_link_libraries(demo_digits_cnn PRIVATE neural)
