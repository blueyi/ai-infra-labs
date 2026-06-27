cmake_minimum_required(VERSION 3.20)
project(CountAndDCEPass)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# 编译成动态加载的 Pass 插件（.so / .dylib）
add_library(CountAndDCEPass MODULE CountAndDCEPass.cpp)
set_target_properties(CountAndDCEPass PROPERTIES CXX_STANDARD 17)
