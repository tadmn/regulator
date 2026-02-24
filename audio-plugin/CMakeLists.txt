cmake_minimum_required(VERSION 3.15)

include(cmake/CPM.cmake)

project(Regulator VERSION 1.0.0)

option(USE_SANITIZER "Build and link with the address sanitizer" OFF)

CPMAddPackage("gh:tadmn/tad-bits#e3a9ef9cc4a9f3bd6a4dff236f755913c77ec124")
include(${tad-bits_SOURCE_DIR}/cmake/compile-options.cmake)

CPMAddPackage("gh:tadmn/choc#2f5b8627708ff4702b4fb6646fddf0c5f4a52007")
CPMAddPackage("gh:tadmn/FastFourier#453106d9d2995aaca8cf53ea556749c1e07bdb05")

find_package(tb-litert CONFIG REQUIRED)
find_package(Boost REQUIRED)

add_subdirectory(${JUCE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/JUCE)

juce_add_plugin(Regulator
  COMPANY_NAME "tadmn"
  PLUGIN_MANUFACTURER_CODE Tadn
  PLUGIN_CODE Rglt
  FORMATS VST3
  PRODUCT_NAME "Regulator"
  COPY_PLUGIN_AFTER_BUILD TRUE)

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
target_sources(Regulator PRIVATE ${SOURCE_FILES})

target_link_libraries(Regulator PRIVATE tad-bits choc FastFourier tb-litert)
target_include_directories(Regulator PRIVATE ${Boost_INCLUDE_DIRS})
target_compile_definitions(Regulator PRIVATE JUCE_VST3_CAN_REPLACE_VST2=0)
add_compiler_warnings(Regulator)