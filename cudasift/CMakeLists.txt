project("cudasift" CXX CUDA)
set(LIBRARY_NAME "cudasift")

set(SOURCE_FILES
    src/cudaImage.cu
    src/cudaSiftD.cu
    src/cudaSiftH.cu
    src/matching.cu
	)
set(HEADER_FILES
    include/cudasift/cudautils.h
    include/cudasift
    )

add_library(${LIBRARY_NAME} ${SOURCE_FILES} ${HEADER_FILES})

target_include_directories(${LIBRARY_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBRARY_NAME} ${CUDA_cudadevrt_LIBRARY}
  )

install(
  TARGETS ${LIBRARY_NAME}
  EXPORT Find${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

