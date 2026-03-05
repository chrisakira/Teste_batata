# Download a whisper model for speech recognition
mkdir -p ~/jarvis/models
cd ~/jarvis/models

# Base model is a good balance of speed/accuracy for real-time
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# For better accuracy (slightly slower):
# wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin