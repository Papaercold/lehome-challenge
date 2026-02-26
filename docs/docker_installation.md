# Docker Installation Guide

This guide provides step-by-step instructions for installing the LeHome Challenge environment using Docker.

## Installation Steps

### 1. Install Docker

```bash
# Install Docker using the convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Post-install steps to run Docker without sudo
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

# Verify the installation
docker run hello-world
```

### 2. Download the Docker Image

```bash
wget https://huggingface.co/datasets/lehome/docker/resolve/main/lehome-challenge.tar.gz
```

> **Note:** Make sure you have sufficient disk space before downloading.

### 3. Load the Docker Image

```bash
docker load -i lehome-challenge.tar.gz
```

### 4. Run and Test the Environment

```bash
# Start the container (adjust flags as needed)
docker run -it lehome-challenge

# Inside the container, activate the environment and verify
cd /opt/lehome-challenge
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none
```

## Next Steps

Now that you have installed the environment, you can:

- [Start Training](training.md)
- [Evaluate Policies](policy_eval.md)
- [Back to README](../README.md)
