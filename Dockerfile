# FROM jupyter/scipy-notebook:latest
FROM custom_python_container:latest
USER root

# Install graphviz
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz graphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the latest versions of these python packages
RUN python -m pip install --upgrade pip && \
    pip install --user numpy scipy pandas bokeh cython networkx graphviz pygraphviz PyQt5 matplotlib opt_einsum

WORKDIR /app

ADD ./src/uncompiled /app/compiled

# Build the Cython files
RUN cd /app/compiled && python /app/compiled/setup.py build_ext --inplace
RUN touch __init__.py