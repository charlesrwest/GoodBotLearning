cmake_minimum_required (VERSION 2.8.3)
PROJECT(GoodBotLearning)

#Add git submodule
#git submodule add https://github.com/philsquared/Catch.git

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules/")

#Set the binaries to be placed in the ./bin/ directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")

#Get c++11
set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED)
find_package(Caffe2 REQUIRED)
find_package(Protobuf REQUIRED)
find_package(CUDA REQUIRED)
find_library(CAFFE2_LIB caffe2)
find_library(CAFFE2_GPU_LIB caffe2_gpu)
find_package(NLOPT REQUIRED)

add_definitions(-DWITH_GPU)

#Tell compiler where to find required libraries
link_directories(/usr/lib/x86_64-linux-gnu/)

include_directories(./src/library/ ./src/executables/unitTests ./src/executables/experimenter ./Catch/single_include ${Caffe2_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${NLOPT_INCLUDE_DIRS})

find_package(Threads)

#Get lists of the source files
FILE(GLOB SOURCEFILES ./src/library/*.cpp ./src/library/*.c)

FILE(GLOB UNIT_TESTS_EXECUTABLE_SOURCE ./src/executables/unitTests/*.cpp ./src/executables/unitTests/*.c)

FILE(GLOB EXPERIMENTER_SOURCE ./src/executables/experimenter/*.cpp ./src/executables/experimenter/*.c)

#Create a libraries
add_library(commonLibrary SHARED  ${SOURCEFILES})


#Tell CMake what executables to make
ADD_EXECUTABLE(unitTests ${UNIT_TESTS_EXECUTABLE_SOURCE}) 

ADD_EXECUTABLE(experimenter ${EXPERIMENTER_SOURCE}) 

target_link_libraries(commonLibrary dl sqlite3 ${CMAKE_THREAD_LIBS_INIT} ${CAFFE2_LIB} ${CAFFE2_GPU_LIB} ${PROTOBUF_LIBRARY} glog ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY} ${NCCL_LIB} ${NLOPT_LIBRARIES}) 

#link libraries to executable
target_link_libraries(unitTests commonLibrary)

target_link_libraries(experimenter commonLibrary)
