cmake_minimum_required(VERSION 3.18 FATAL_ERROR)



ConfigureExecutable(test_components "${CMAKE_CURRENT_SOURCE_DIR}/src/test_components.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(test_key_val_pair "${CMAKE_CURRENT_SOURCE_DIR}/src/test_key_val_pair.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(speed_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/speed_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(kmer_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/kmer_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(power_2_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/power_2_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

#ConfigureExecutable(separable_test "${CMAKE_CURRENT_SOURCE_DIR}/src/separable_test.cpp" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(bucket_speed_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/bucket_speed_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(test_cg_variations "${CMAKE_CURRENT_SOURCE_DIR}/src/test_cg_variations.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(test_mhm_tcf "${CMAKE_CURRENT_SOURCE_DIR}/src/test_mhm_tcf.cu" "${POGGERS_TESTS_BINARY_DIR}")



ConfigureExecutable(delete_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/delete_tests.cu" "${POGGERS_TESTS_BINARY_DIR}")

ConfigureExecutable(sawtooth_test "${CMAKE_CURRENT_SOURCE_DIR}/src/sawtooth_test.cu" "${POGGERS_TESTS_BINARY_DIR}")

