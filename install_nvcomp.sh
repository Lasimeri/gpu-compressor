#!/bin/bash
set -e

echo "🚀 Installing nvCOMP for GPU compression"
echo ""
echo "nvCOMP must be downloaded manually from NVIDIA Developer:"
echo "https://developer.nvidia.com/nvcomp-download"
echo ""
echo "After downloading nvcomp-linux-x86_64-X.X.X-archive.tar.xz:"
echo ""
echo "1. Extract the archive:"
echo "   tar -xf nvcomp-linux-x86_64-*-archive.tar.xz"
echo ""
echo "2. Install headers and libraries:"
echo "   sudo cp -r nvcomp-linux-x86_64-*-archive/include/* /usr/local/include/"
echo "   sudo cp -r nvcomp-linux-x86_64-*-archive/lib/* /usr/local/lib/"
echo "   sudo ldconfig"
echo ""
echo "3. Verify installation:"
echo "   ls /usr/local/include/nvcomp.h"
echo "   ls /usr/local/lib/libnvcomp*"
echo ""
echo "4. Build the project:"
echo "   cargo build --release"
echo ""

# Try to install from AUR if on Arch Linux
if command -v yay &> /dev/null; then
    echo "Detected Arch Linux with yay - trying AUR package..."
    yay -S nvcomp || echo "AUR package not available, use manual installation above"
fi
