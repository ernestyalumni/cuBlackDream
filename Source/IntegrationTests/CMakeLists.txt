ENABLE_TESTING()

SET(TEST_SOURCES
  DataWrangling/Kaggle/ParseDigitRecognizer_tests.cpp
  )

ADD_EXECUTABLE(RunIntegrationTests
  ${TEST_SOURCES}
  )

TARGET_LINK_LIBRARIES(RunIntegrationTests
  DataWrangling
  UtilitiesCU
  gtest_main
  )

INCLUDE(GoogleTest)
gtest_discover_tests(RunIntegrationTests)

SET_TARGET_PROPERTIES(RunIntegrationTests
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")