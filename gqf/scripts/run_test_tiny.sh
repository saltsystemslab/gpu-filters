#!/bin/bash
echo "running tests for $1" 
mkdir -p $2/$1
./test -n 22 -d $1 -o $2/$1/22
./test -n 24 -d $1 -o $2/$1/24 
./test -n 26 -d $1 -o $2/$1/26 