FROM condaforge/miniforge3:latest
WORKDIR /work
COPY environment.yml .
RUN mamba env create -f environment.yml && mamba clean -afy
COPY . .
SHELL ["conda", "run", "-n", "spatialrl", "/bin/bash", "-c"]
CMD ["conda", "run", "--no-capture-output", "-n", "spatialrl", "make", "all"]
