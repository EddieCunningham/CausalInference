docker build -t causalinference -f Dockerfile .
docker run -it --rm -v "$(pwd)":/app/host causalinference /bin/bash