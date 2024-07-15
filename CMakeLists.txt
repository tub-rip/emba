cmake_minimum_required(VERSION 3.15)
project(emba)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "-fopenmp -O4 ${CMAKE_CXX_FLAGS}")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(thirdparty/basalt-headers/thirdparty/Sophus)
include_directories(thirdparty/basalt-headers/thirdparty/cereal/include)
include_directories(thirdparty/basalt-headers/include)

set(SOURCES
   src/emba/node.cpp
   src/emba/emba.cpp
   src/emba/solver.cpp
   src/emba/model.cpp
   src/image_rec/laplace.cpp
   src/image_rec/poisson_reconstruction.cpp
   src/utils/pose_manager.cpp
   src/utils/trajectory.cpp
   src/utils/event_pano_warper.cpp
   src/utils/rosbag_loading.cpp
   src/utils/eigen_utils.cpp
   src/utils/image_utils.cpp
   )

# Node
cs_add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
   fftw3 fftw3_threads
)

cs_export()
