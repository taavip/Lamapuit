# Dockerfile for Lamapuit project with Conda and Mamba
FROM continuumio/miniconda3:latest

# Install mamba for faster environment solving
RUN conda install -y mamba -c conda-forge

# Install system build tools needed for pip builds (gcc, make, etc.)
RUN apt-get update && apt-get install -y build-essential gcc g++ libffi-dev && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy environment file
COPY environment.yml ./

# Create the environment
RUN mamba env create -f environment.yml

# Activate environment by default (for interactive shells)
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate cwd-detect" >> ~/.bashrc

# Set default command but allow override
CMD ["/bin/bash", "--login"]
