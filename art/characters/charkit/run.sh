#!/usr/bin/env bash
# Build a character: charkit/run.sh <character folder> [build options]
# Prints timings, scores and errors; the full Blender log goes to $TEMP.
CHAR="$1"; shift
cd "$(dirname "$0")/../$CHAR" || exit 1
BLENDER="${BLENDER:-/c/Program Files/Blender Foundation/Blender 4.5/blender.exe}"
LOG="${TEMP:-/tmp}/${CHAR}_build.log"
"$BLENDER" -b --factory-startup -P scripts/build.py -- "$@" > "$LOG" 2>&1
grep -n "IoU\|Error\|Traceback\|RuntimeWarning\|built in\|done in\|mesh stats" "$LOG" | head -30
grep -B2 -A12 "Traceback" "$LOG" | head -40
echo "log: $LOG"
