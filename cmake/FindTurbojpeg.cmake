include(CheckCXXSourceCompiles)
include(FindPackageHandleStandardArgs)

find_path(Turbojpeg_INCLUDE_DIR NAMES turbojpeg.h)
find_library(Turbojpeg_LIBRARY NAMES libturbojpeg.a)


find_package_handle_standard_args(turbojpeg DEFAULT_MSG Turbojpeg_LIBRARY Turbojpeg_INCLUDE_DIR)

set(CMAKE_REQUIRED_INCLUDES ${Turbojpeg_INCLUDE_DIR})
set(CMAKE_REQUIRED_LIBRARIES ${Turbojpeg_LIBRARY})
message(STATUS ${CMAKE_REQUIRED_LIBRARIES})
check_cxx_source_compiles("#include <turbojpeg.h>\nint main() { tjhandle handle = tjInitDecompress(); return 0;}" TurboJpeg_COMPILES)

if (NOT TurboJpeg_COMPILES)
    message(FATAL_ERROR "Failed to compile sample turbojpeg code")
else()
    message(STATUS "Found working turbojpeg at ${TurboJpeg_INCLUDE_DIR}")
    add_library(turbojpeg::turbojpeg STATIC IMPORTED)
    set_target_properties(turbojpeg::turbojpeg PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${Turbojpeg_INCLUDE_DIR}
        IMPORTED_LOCATION ${Turbojpeg_LIBRARY})
endif()

unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)
