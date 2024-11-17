#!/bin/bash

# Usage: ./setup.sh [--force|-f]
# Options:
#   --force, -f    Remove existing models and clone again

# Initialize FORCE flag
FORCE=false

export DOCKER_DEFAULT_PLATFORM=linux/arm64

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE=true
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: ./setup.sh [--force|-f]"
            exit 1
            ;;
    esac
    shift
done

# Create models and logs directories if they don't exist
mkdir -p models
mkdir -p logs

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null
then
    echo "git-lfs could not be found. Please install it before running this script."
    exit 1
fi

# Navigate to models directory
cd models

# Remove existing cloned repositories if FORCE is true
if [ "$FORCE" = true ]; then
    echo "Removing existing models..."
    rm -rf layoutlmv3-base layoutlmv3-base-finetuned-rvlcdip

    git clone https://huggingface.co/microsoft/layoutlmv3-base
    git clone https://huggingface.co/gordonlim/layoutlmv3-base-finetuned-rvlcdip
fi

# Navigate back to the project root
cd ..

# Start Docker services
docker-compose up --build --abort-on-container-failure