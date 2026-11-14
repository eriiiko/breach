#!/usr/bin/env bash
# Build wrapper: ./run.sh [build.py options]; prints scores, errors and the mismatch table.
cd "$(dirname "$0")"
BLENDER="${BLENDER:-/c/Program Files/Blender Foundation/Blender 4.5/blender.exe}"
LOG="${TEMP:-/tmp}/space_marine_build.log"
"$BLENDER" -b --factory-startup -P scripts/build.py -- "$@" > "$LOG" 2>&1
grep -n "IoU\|Error\|Traceback\|RuntimeWarning\|built in\|done in\|mesh stats" "$LOG" | head -30
echo "log: $LOG"
