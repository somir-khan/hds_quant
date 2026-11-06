#!/bin/bash
# Source from the project root before starting Python. .env is trusted Bash.
# Disable tracing so credentials cannot be printed by shell debug output.
set +x
if [ -f .env ]; then
    # Export assignments to every Python child, even without `export` in .env.
    case $- in
        *a*) source .env ;;
        *) set -a; source .env; set +a ;;
    esac
fi
