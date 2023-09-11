# https://cmake.org/cmake/help/latest/command/add_library.html#interface-libraries
# Creates an interface library; an INTERFACE library target does not compile
# sources and doesn't produce a library artiface on disk. However, it may have
# properties set on it and may be installed and exported.
ADD_LIBRARY(cudnn_frontend INTERFACE)

# https://cmake.org/cmake/help/latest/command/target_include_directories.html#command:target_include_directories
TARGET_INCLUDE_DIRECTORIES(
  cudnn_frontend INTERFACE
  # https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html#genex:BUILD_INTERFACE
  # Content when property is exported using export() o r when target used by
  # another target in same buildsystem.
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include>
  # https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html#genex:INSTALL_INTERFACE
  # Content when property is exported using install(EXPORT)
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )