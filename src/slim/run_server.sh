#!/usr/bin/env bash
NUM_WORKER=2
BIND_ADDR=0.0.0.0:5001
gunicorn -w ${NUM_WORKER} -b ${BIND_ADDR} -p gunicorn.pid classify_images:app > engine.log 2>&1 &
