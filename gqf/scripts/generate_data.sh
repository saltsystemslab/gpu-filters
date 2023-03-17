#!/bin/bash
echo "Starting performance tests"
./scripts/run_all_tests.sh "results"
echo "Done generating data, aggregating results..."
mkdir -p results/aggregate
python3 scripts/aggregate_local_data.py
cd latex_files
pdflatex -output-directory .. artifact_figures.tex
cd ..
echo "Done"
