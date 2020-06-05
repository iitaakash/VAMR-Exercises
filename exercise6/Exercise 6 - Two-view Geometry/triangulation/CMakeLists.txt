cmake_minimum_required(VERSION 3.0)
project( triangulation )
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# add_executable( main main.cpp Stereo.cpp)
# target_link_libraries( main ${OpenCV_LIBS} )
# target_link_libraries (main Eigen3::Eigen)

# add_executable( triangualte_test triangualte_test.cpp Triangulate.cpp)
# target_link_libraries( triangualte_test ${OpenCV_LIBS} )
# target_link_libraries (triangualte_test Eigen3::Eigen)

add_executable( 8_point_test 8_point_test.cpp Triangulate.cpp)
target_link_libraries( 8_point_test ${OpenCV_LIBS} )
target_link_libraries (8_point_test Eigen3::Eigen)