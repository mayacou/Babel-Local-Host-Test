#!/bin/bash

# Define variables
POD_IP="ssh.runpod.io"  # The hostname for the RunPod pod
POD_USER="pkrc5k47d2f4c1-64411192"  # Your username for the pod
GIT_REPO="git@github.com:mayacou/Babel-Local-Host-Test.git"  # Your repo URL
BRANCH_NAME="eval-branch-paul"  # The branch you want to checkout
WORKSPACE_DIR="workspace"
HF_TOKEN="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"  # Your Hugging Face token
DOCKER_IMAGE_NAME="runpod_image"  # Your Docker image name

# SSH into the pod and run Docker
ssh -i $POD_USER@$POD_IP << EOF

  # Step 2: Create the workspace directory and clone the repo
  mkdir -p ~/$WORKSPACE_DIR
  cd ~/$WORKSPACE_DIR
  git clone $GIT_REPO  # Clone the repository

  # Step 3: Switch to the desired branch
  cd $(basename $GIT_REPO .git)  # Navigate into the cloned repo
  git checkout $BRANCH_NAME  # Checkout the specific branch

  # Step 4: Set Hugging Face authentication
  echo $HF_TOKEN > ~/.huggingface/token

  # Step 5: Build the Docker image
  docker build -t $DOCKER_IMAGE_NAME .  # Build the Docker image

  # Step 6: Run the Docker container
  docker run --rm $DOCKER_IMAGE_NAME  # Run the Docker container

EOF
