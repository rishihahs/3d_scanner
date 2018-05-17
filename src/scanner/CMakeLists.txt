project(scanner)

cmake_minimum_required (VERSION 3.0)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package (Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io visualization)

add_definitions(${PCL_DEFINITIONS})

add_executable(scanner
  src/scanner.cpp
  src/PtCloudReconstructor.cpp
  src/ModelPlaque.cpp
  src/PlaqueLocalizer.cpp
  src/ScannerUtils.cpp
)

include_directories(scanner
  include
  ../mapping_calibration/include
  ../ar_track_alvar/include
  /usr/include/openni2
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
    ../mapping_calibration
    ../ar_track_alvar
	/usr/lib/x86_64-linux-gnu/
)

target_link_libraries(scanner
  ar_track_alvar
  mapping_calibration
  OpenNI2
  ${OpenCV_LIBRARIES}
  ${PCL_COMMON_LIBRARIES}
  ${PCL_IO_LIBRARIES}
  ${PCL_VISUALIZATION_LIBRARIES}
)

set_property(TARGET scanner PROPERTY CXX_STANDARD 11)
