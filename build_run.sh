# Install system dependencies (CachyOS/Arch)
sudo pacman -S base-devel clang cmake pkg-config alsa-lib pipewire

# Build the project
cd jarvis-assistant
cargo build --release

# Run with environment variables
WHISPER_MODEL_PATH=~/jarvis/models/ggml-base.en.bin \
PIPER_MODEL_PATH=~/jarvis/voices/en_US-lessac-medium.onnx \
OLLAMA_URL=http://localhost:11434 \
OLLAMA_MODEL=mistral:7b-instruct-v0.3-q5_K_M \
SENSOR_API_URL=http://localhost:8080 \
RUST_LOG=info \
cargo run --release