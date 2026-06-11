#!/bin/bash
#SBATCH -J data_download_cpu     # Job name
#SBATCH -o download.%j.out       # Output file (%j = job ID)
#SBATCH -t 24:00:00              # Wall time (1 hour)
#SBATCH -N 1                     # Number of nodes
#SBATCH -n 1                     # Total tasks (4 per node)
#SBATCH -p gg                    # GPU partition (modify as needed)
#SBATCH -A ASC26008              # Project allocation

gsutil -m cp -r gs://gresearch/robotics/droid $SCRATCH/.cache/droid/
