
add_executable(cudasift_test mainSift.cpp)
target_link_libraries(cudasift_test cudasift ${OpenCV_LIBRARIES})
target_include_directories(cudasift_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cudasift_bulk bulkDetect.cpp)
target_link_libraries(cudasift_bulk cudasift ${OpenCV_LIBRARIES})
target_include_directories(cudasift_bulk  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


install(
  TARGETS cudasift_test cudasift_bulk
  EXPORT Find${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

