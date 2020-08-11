# Docker Spark Cluster (v2.4.5) for Windows
A fully automated way to build Docker images and boot up a Spark cluster. Basic setup includes
* 1 Spark Master
* 1 Spark Submit
* 2 Spark Workers

To increase the number of workers, you simply have to modify `docker-compose.yml`

To change the hardware settings of the workers, modify `env\spark-worker-env.sh`

## Requirements
* Docker for Windows with Linux Containers
* Disabled TLS security

## How-to
* Build the images by executing `build-docker.bat`
* Modify `env\spark-worker-env.sh` to fit your computing power needs
* Start the Spark cluster by executing `start-spark-cluster.bat`