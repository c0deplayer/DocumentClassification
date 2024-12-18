#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 [-f|--force] [-d|--detach]  [-p|--password PASSWORD] [-r|--rebuild]"
    echo "Options:"
    echo "  -f, --force      Remove existing models and clone again"
    echo "  -d, --detach     Run in detached mode"
    echo "  -p, --password   Set PostgreSQL password"
    echo "  -r, --rebuild    Rebuild docker images"
    echo "  -h, --help       Display this help message"
    exit 1
}

# Initialize FORCE flag
FORCE=false
POSTGRES_PASSWORD=""
REBUILD=false
DETACH=false

# Check for password in .env
if [ -z "$POSTGRES_PASSWORD" ] && [ -f ".env" ]; then
    source .env
    POSTGRES_PASSWORD=${DB_PASSWORD:-$POSTGRES_PASSWORD}
fi

export POSTGRES_PASSWORD

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force) FORCE=true ;;
        -p|--password) POSTGRES_PASSWORD="$2"; shift ;;
        -r|--rebuild) REBUILD=true; docker-compose down --volumes && docker-compose build --no-cache ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Check if any docker containers exist and stop them
if [ "$(docker ps -q)" ]; then
    echo "Stopping existing containers..."
    docker-compose down --volumes
fi

# Create models and logs directories if they don't exist
mkdir -p models
mkdir -p logs

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null
then
    echo "git-lfs could not be found. Please install it before running this script."
    exit 1
fi

# Remove existing cloned repositories if FORCE is true
if [ "$FORCE" = true ]; then
    cd models

    echo "Removing existing models..."
    rm -rf layoutlmv3-base layoutlmv3-base-finetuned-rvlcdip

    git clone https://huggingface.co/microsoft/layoutlmv3-base
    git clone https://huggingface.co/gordonlim/layoutlmv3-base-finetuned-rvlcdip

    cd ..
fi

if [ "$DETACH" = true ]; then
    echo "Starting services in detached mode..."
    docker-compose up -d
elif [ "$REBUILD" = false ]; then
    echo "Building and starting services..."
    docker-compose up --build
else
    echo "Starting services..."
    docker-compose up
fi
