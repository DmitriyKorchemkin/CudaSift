
add_executable(cudasift_test mainSift.cpp)
target_link_libraries(cudasift_test cudasift ${OpenCV_LIBRARIES})

install(
  TARGETS cudasift
  EXPORT Find${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

