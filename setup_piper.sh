# Install Piper
sudo pacman -S piper-tts
# Or from AUR if not in main repos:
# yay -S piper-tts-bin

# Download a voice model (e.g., a good English voice)
mkdir -p ~/jarvis/voices
cd ~/jarvis/voices
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/voice-en_US-lessac-medium.tar.gz
tar -xzf voice-en_US-lessac-medium.tar.gz

# Test it
echo "Hello, I am Jarvis, your local assistant." | \
  piper --model en_US-lessac-medium.onnx --output_file test.wav
aplay test.wav