#!/usr/bin/env bash
set -e
# Ensure the same Python version and Vermin version are used
pip install vermin==1.6.0
# Use find to only pass Python files
find . -type f -name '*.py' -print0 | xargs -0 vermin --target=3.12 \
       --violations \
       --backport argparse \
       --backport asyncio \
       --backport configparser \
       --backport dataclasses \
       --backport enum \
       --backport importlib \
       --backport ipaddress \
       --backport mock \
       --backport typing \
       --backport typing_extensions \
       --no-parse-comments \
       --eval-annotations
