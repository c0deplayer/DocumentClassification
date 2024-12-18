#!/bin/bash

# Selected model.
MODEL=llama3.2:3b
# MODEL=gemma2:2b

# Start Ollama in the background.
echo "Starting Ollama server..."
ollama serve &
SERVE_PID=$!

echo "Waiting for Ollama server to be active..."
while ! ollama list | grep -q 'NAME'; do
  sleep 1
done

echo "🔴 Retrieve ${MODEL} model..."
ollama pull ${MODEL}
echo "🟢 Done!"

# Preload the model.
echo "🔴 Preload ${MODEL} model..."
ollama run ${MODEL} ""
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $SERVE_PID
