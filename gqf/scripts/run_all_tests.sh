#!/bin/bash
echo "Starting all tests"
./scripts/run_test.sh gqf $1
./scripts/run_test.sh point $1
./scripts/run_test_tiny.sh sqf $1
./scripts/run_test_tiny.sh rsqf $1
./scripts/run_test.sh bloom $1