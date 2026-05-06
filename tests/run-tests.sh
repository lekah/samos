#!/bin/bash
files=`ls test_*.py`
for file in $files; do
    echo -e "\n! running $file"
    python $file
done
