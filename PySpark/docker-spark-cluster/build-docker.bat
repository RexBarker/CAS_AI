docker build -t spark-base:2.4.5 docker/base
docker build -t spark-master:2.4.5 docker/spark-master
docker build -t spark-worker:2.4.5 docker/spark-worker
docker build -t spark-submit:2.4.5 docker/spark-submit