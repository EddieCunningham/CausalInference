nvidia-docker build -t causalinference -f DockerfileTF .
nvidia-docker run -it --rm -v "$(pwd)":/app/host causalinference /bin/bash