# Start with a base image of Ubuntu
FROM ubuntu:20.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    python3-pip \
    python3-dev \
    libgsl-dev \
    libgsl23

# Download and install ViennaRNA
RUN wget https://www.tbi.univie.ac.at/RNA/download/ubuntu/ubuntu_23_04/python3-rna_2.6.4-1_amd64.deb && \
    dpkg -i python3-rna_2.6.4-1_amd64.deb && \
    rm python3-rna_2.6.4-1_amd64.deb

# Install arnie
RUN pip3 install -r requirements.txt

# Set the default command for the container, open a bash shell
CMD ["bash"]