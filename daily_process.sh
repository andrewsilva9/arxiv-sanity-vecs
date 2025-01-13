#!/bin/bash
source ~/.virtualenvs/flaskpy3/bin/activate

cd ~/arxiv-sanity-vecs/

# Run the paper retriever
python3 arxiv_daemon.py --num 20000

python3 seq_embeddings.py

echo "Starting Flask app"
export FLASK_APP=serve.py
flask run & # Start the Flask app in the background
FLASK_PID=$! # Get the process ID of the Flask app

# Keep Flask app running for 3 hours
sleep 10800

# Kill the Flask app after 3 hours
echo "Stopping Flask app"
kill $FLASK_PID
