#!/bin/sh

# Get the path of this file
# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

docker build -t causalinference .
docker run -it --rm -v "$(pwd)":/app/host causalinference /bin/bash