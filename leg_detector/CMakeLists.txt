cmake_minimum_required(VERSION 2.8.3)
project(leg_detector)

# look for bfl (Bayesian Filtering Library)
find_package(PkgConfig REQUIRED)
pkg_check_modules(BFL REQUIRED orocos-bfl)

include_directories(${BFL_INCLUDE_DIRS})
link_directories(${BFL_LIBRARY_DIRS})

# Look for Bullet
pkg_check_modules(BULLET bullet)
include_directories(${BULLET_INCLUDE_DIRS})

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  std_srvs
  sensor_msgs
  laser_geometry
  tf
  visualization_msgs
  people_msgs
  people_tracking_filter
  image_geometry
  dynamic_reconfigure
)

## dynamic reconfigure config
generate_dynamic_reconfigure_options(
  cfg/LegDetector.cfg
)

## Specify additional locations of header files
include_directories(
  include ${catkin_INCLUDE_DIRS}
)

catkin_package(INCLUDE_DIRS include
  LIBRARIES leg_detector_lib
  CATKIN_DEPENDS people_msgs sensor_msgs std_msgs geometry_msgs visualization_msgs)

## Declare a cpp executable
add_executable(leg_detector 
               src/laser_processor.cpp
               src/leg_detector.cpp 
               src/calc_leg_features.cpp)

## Add cmake target dependencies of the executable/library
add_dependencies(leg_detector people_msgs_gencpp ${${PROJECT_NAME}_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(leg_detector
   ${catkin_LIBRARIES} ${BFL_LIBRARIES} ${BULLET_LIBRARIES}
)
set_target_properties(leg_detector PROPERTIES COMPILE_OPTIONS "-std=c++11")

## Declare a cpp library
add_library(leg_detector_lib 
            src/laser_processor.cpp
            src/leg_detector_lib.cpp 
            src/calc_leg_features.cpp
            src/saved_feature.cpp)

## Add cmake target dependencies of the executable/library
add_dependencies(leg_detector_lib people_msgs_gencpp ${${PROJECT_NAME}_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(leg_detector_lib
   ${catkin_LIBRARIES} ${BFL_LIBRARIES} ${BULLET_LIBRARIES}
)
set_target_properties(leg_detector_lib PROPERTIES COMPILE_OPTIONS "-std=c++11")

install(TARGETS
    leg_detector
    leg_detector_lib
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES config/trained_leg_detector.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
