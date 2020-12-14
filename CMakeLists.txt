cmake_minimum_required(VERSION 3.15)
project(TAJIVEdecomp)

set(CMAKE_CXX_STANDARD 20)


set(EIGEN_TENSOR_DIR "/home/virginie/Desktop/VITALE/Tesi/Codice/TAJIVEc++/dependecies/unsupported")
if(NOT EXISTS ${EIGEN_TENSOR_DIR})
    message(FATAL_ERROR "Please check that the set Eigen Unsupported directory is valid!")
endif()

include_directories(${EIGEN_TENSOR_DIR})

set(EIGEN_DIR "/home/virginie/Desktop/VITALE/Tesi/Codice/TAJIVEc++/dependecies/Eigen")
if(NOT EXISTS ${EIGEN_DIR})
    message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
endif()

#[[set(ARMADILLO_DIR "/home/virginie/Desktop/VITALE/Tesi/Codice/TAJIVEc++/dependecies/armadillo")
if(NOT EXISTS ${ARMADILLO_DIR})
    message(FATAL_ERROR "Please check that the set Armadillo directory is valid!")
endif()]]

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

include_directories(${EIGEN_TENSOR_DIR})

include_directories(${EIGEN_TENSOR_DIR})
add_executable(TAJIVEdecomp TAJIVEmain.cpp include/TensorDyn.h src/TensorDyn.cpp include/TAJIVE.h src/TAJIVE.cpp )

target_link_libraries(TAJIVEdecomp ${ARMADILLO_LIBRARIES}) # <<<