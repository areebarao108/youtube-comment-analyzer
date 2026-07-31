#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 289400855433.dkr.ecr.us-east-1.amazonaws.com
echo "Pulling Docker image..."
docker pull 289400855433.dkr.ecr.us-east-1.amazonaws.com/yt-chrome-plugin:latest
echo "Checking for existing container..."
if [ "$(docker ps -q -f name=yt-pligin)" ]; then
    echo "Stopping existing container..."
    docker stop yt-plugin
fi

if [ "$(docker ps -aq -f name=yt-plugin)" ]; then
    echo "Removing existing container..."
    docker rm yt-plugin
fi

echo "Starting new container..."
docker run -d -p 80:5000 --name cyt-plugin 289400855433.dkr.ecr.us-east-1.amazonaws.com/yt-chrome-plugin:latest

echo "Container started successfully."