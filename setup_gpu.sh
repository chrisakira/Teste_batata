# CachyOS usually has good AMD support out of the box
# Install ROCm for GPU-accelerated LLM inference
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk

# Verify GPU is detected
rocminfo | grep -i "gfx"
# Your RX6800 should show as gfx1030

# Install Vulkan support (for whisper.cpp)
sudo pacman -S vulkan-radeon vulkan-tools
vulkaninfo | grep "deviceName"