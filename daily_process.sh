#!/bin/bash
__conda_setup="$('/Users/andrew_silva2/miniforge3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/andrew_silva2/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/Users/andrew_silva2/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/andrew_silva2/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate flaskpy3

cd ~/Documents/code/arxiv-sanity-vecs/

# Run the paper retriever
python arxiv_daemon.py --num 2000

python seq_embeddings.py

echo "Starting Flask app"
export FLASK_APP=serve.py
flask run & # Start the Flask app in the background
FLASK_PID=$! # Get the process ID of the Flask app

# Keep Flask app running for 3 hours
sleep 10800

# Kill the Flask app after 3 hours
echo "Stopping Flask app"
kill $FLASK_PID
