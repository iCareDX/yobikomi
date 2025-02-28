#!/usr/bin/env bash

cd "${0%/*}"
./venv_sys/bin/python -u ./test_llm.py
