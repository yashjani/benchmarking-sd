#!/bin/bash

# Install necessary packages
apt-get update
apt-get install -y awscli curl unzip git python3-pip

# Set the HOME environment variable
export HOME=/home/ubuntu

# Configure Git credentials
TOKEN="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"
REPO_URL="https://github.com/yashjani/benchmarking-sd.git" # Replace with your repo's URL
git config --global credential.helper 'cache --timeout=3600'
CREDENTIALS_FILE=$(mktemp)
echo "url=$REPO_URL" > $CREDENTIALS_FILE
echo "username=$TOKEN" >> $CREDENTIALS_FILE
echo "password=x-oauth-basic" >> $CREDENTIALS_FILE
git credential-cache store < $CREDENTIALS_FILE
rm $CREDENTIALS_FILE

# Clone the repository
git clone $REPO_URL /home/ubuntu/benchmarking-sd

# Change ownership of the repository directory
chown -R ubuntu:ubuntu /home/ubuntu/benchmarking-sd

# Navigate to the repository directory
cd /home/ubuntu/benchmarking-sd

# Configure Git safe directory
git config --global --add safe.directory /home/ubuntu/benchmarking-sd

# Pull the latest changes from the repository
git pull

# Install required Python packages
pip3 install -r requirements.txt

# Run the benchmark script
/usr/bin/python3 /home/ubuntu/benchmarking-sd/benchmark.py --server_name "p5.48xlarge" --model_name "CompVis/stable-diffusion-v1-4" --ondemand_cost 98.320 --spot_cost 33.860 --reserved_one_year_cost 0 --reserved_three_year_cost 43.157

# Commit and push any changes
git add -A
git commit -m "p5.48xlarge"
git push origin main