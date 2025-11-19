nohup ./run_docker_job.sh > training.log 2>&1 &
export PYTHONPATH="$PWD:$PYTHONPATH"