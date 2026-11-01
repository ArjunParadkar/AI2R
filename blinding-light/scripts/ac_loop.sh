#!/usr/bin/env bash
# AC-aware runner: runs a resumable command only while on AC power (polls every 60 s); stops the child on unplug; retries until exit code 0.
# usage: scripts/ac_loop.sh <cmd...>
cd "$(dirname "$0")/.."
ac() { [ "$(cat /sys/class/power_supply/ACAD/online 2>/dev/null)" = 1 ]; }
while true; do
  until ac; do sleep 60; done
  "$@" & pid=$!
  while kill -0 $pid 2>/dev/null; do ac || { kill $pid 2>/dev/null; wait $pid 2>/dev/null; break; }; sleep 20; done
  if ! kill -0 $pid 2>/dev/null; then wait $pid 2>/dev/null; rc=$?; [ $rc = 0 ] && exit 0; ac && sleep 120; fi
done
