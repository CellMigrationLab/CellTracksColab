#!/usr/bin/env bash
# Helper to run a command, capture its output, and build a context-rich log.
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: run_with_context.sh <log-path> <step-label> <script-path> <command...>" >&2
  exit 2
fi

LOG_PATH="$1"
shift
STEP_LABEL="$1"
shift
SCRIPT_PATH="$1"
shift

WORKFLOW_PATH="${WORKFLOW_PATH:-.github/workflows/update_on_notebook_change.yml}"
INSTRUCTIONS=$'You are explaining a GitHub Actions error to a non-programmer maintaining a LabConstrictor-based repository.\nAlways start with a short summary of what failed and in which workflow step.\nWhen the log points to dependency or requirements issues, remind them that each notebook can ship a requirements.yaml, LabConstrictor merges those into a single requirements.txt/environment.yaml, and a shared Python version plus common package list is required so the generated installer works for every notebook.\nIf the problem clearly is not about dependencies, explicitly say so and direct them to open/consult an issue at https://github.com/CellMigrationLab/LabConstrictor/issues.\nStructure your explanation as:\n1. One or two sentences describing what went wrong and where.\n2. The most likely cause in simple language (include the requirements context above when relevant).\n3. Clear next steps to fix or verify (no code unless essential).\n\nThis file contains: (a) the error log, (b) the workflow YAML, (c) the Python script.\n'

TMP_FILE="$(mktemp)"

set +e
"$@" 2>&1 | tee "$TMP_FILE"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ]; then
  mkdir -p "$(dirname "$LOG_PATH")"
  {
    printf '%s\n' "$INSTRUCTIONS"
    printf 'There has been an error while running %s, this is the error message:\n' "$STEP_LABEL"
    cat "$TMP_FILE"
    printf '\nThis is the GitHub action workflow:\n'
    cat "$WORKFLOW_PATH"
    printf '\nThis is the %s script:\n' "$STEP_LABEL"
    cat "$SCRIPT_PATH"
  } > "$LOG_PATH"
else
  : # No log file on success
fi

rm -f "$TMP_FILE"
exit "$STATUS"
