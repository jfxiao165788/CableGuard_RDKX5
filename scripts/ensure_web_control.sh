#!/usr/bin/env bash
set -euo pipefail

export CABLE_WORKSPACE_DIR="${CABLE_WORKSPACE_DIR:-/home/sunrise/cable1}"
LOG_FILE="${LOG_FILE:-/tmp/web.log}"
LOCK_FILE="${LOCK_FILE:-/tmp/cable1_web_control.lock}"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  exit 0
fi

if ss -ltn 2>/dev/null | grep -q ':8010 '; then
  exit 0
fi

source /opt/ros/humble/setup.bash
source "$CABLE_WORKSPACE_DIR/install/setup.bash"

setsid ros2 run motor_control_pkg web_control_node >>"$LOG_FILE" 2>&1 < /dev/null &
