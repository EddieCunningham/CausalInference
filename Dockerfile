FROM jupyter/scipy-notebook:latest
WORKDIR /app
USER root

ADD ./src/uncompiled /app/compiled

# Install graphviz
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the latest versions of these python packages
RUN python -m pip install --upgrade pip && \
    pip install --user numpy scipy pandas bokeh cython networkx graphviz

# Build the Cython files
RUN cd /app/compiled && python /app/compiled/setup.py build_ext --inplace
RUN touch __init__.py