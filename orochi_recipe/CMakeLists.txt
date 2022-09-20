

cmake_minimum_required(VERSION 3.0.0)
project(cmake_wrapper)

include(conanbuildinfo.cmake)
conan_basic_setup()

message(${CMAKE_SOURCE_DIR})

add_library(orochi STATIC source_subfolder/Orochi/Orochi.cpp source_subfolder/Orochi/OrochiUtils.cpp source_subfolder/contrib/cuew/src/cuew.cpp source_subfolder/contrib/hipew/src/hipew.cpp)
target_include_directories(orochi PUBLIC ${CMAKE_SOURCE_DIR}/source_subfolder)

install(TARGETS orochi DESTINATION lib)
install (
    DIRECTORY ${CMAKE_SOURCE_DIR}/source_subfolder/Orochi
    DESTINATION include
    FILES_MATCHING PATTERN "*.h*")