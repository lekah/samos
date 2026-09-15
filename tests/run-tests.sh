#!/bin/bash
# Run every test file and report which ones failed.
# Plain unittest scripts, so each file is a separate python run; the
# loop keeps going after a failure so one broken file does not hide
# the state of the rest, and the exit status is non-zero if any failed.

cd "$(dirname "$0")" || exit 1

failed=()
for file in test_*.py; do
    echo -e "\n! running $file"
    # -u: unbuffered stdout. Without it, print() output (buffered,
    # since this isn't a terminal) and unittest's own OK/FAILED verdict
    # (unbuffered, on stderr) can come out in the wrong order, burying
    # the verdict in the middle of a file's printed output instead of
    # at the end where it's expected.
    python -u "$file" || failed+=("$file")
done

if [ ${#failed[@]} -gt 0 ]; then
    echo -e "\nFAILED: ${failed[*]}"
    exit 1
fi
echo -e "\nAll test files passed."
