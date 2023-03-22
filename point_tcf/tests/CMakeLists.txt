cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

ConfigureExecutable(speed_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/speed_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(test_cg_variations "${CMAKE_CURRENT_SOURCE_DIR}/src/test_cg_variations.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(test_mhm_tcf "${CMAKE_CURRENT_SOURCE_DIR}/src/test_mhm_tcf.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(delete_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/delete_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(sawtooth_test "${CMAKE_CURRENT_SOURCE_DIR}/src/sawtooth_test.cu" "${POGGERS_TESTS_BINARY_DIR}")

