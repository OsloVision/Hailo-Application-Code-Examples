#!/bin/bash

# FFmpeg Capture Service Installation Script

SERVICE_NAME="ffmpeg-capture"
SERVICE_FILE="$SERVICE_NAME.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "Installing FFmpeg Capture Service..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run this script as root (use sudo)"
    exit 1
fi

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: $SERVICE_FILE not found in current directory"
    exit 1
fi

# Copy service file to systemd directory
echo "Copying service file to $SYSTEMD_DIR..."
cp "$SERVICE_FILE" "$SYSTEMD_DIR/"

# Set proper permissions
chmod 644 "$SYSTEMD_DIR/$SERVICE_FILE"

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service to start on boot
echo "Enabling service to start on boot..."
systemctl enable "$SERVICE_NAME"

echo "Service installed successfully!"
echo ""
echo "To start the service now: sudo systemctl start $SERVICE_NAME"
echo "To check service status: sudo systemctl status $SERVICE_NAME"
echo "To view logs: sudo journalctl -u $SERVICE_NAME -f"
echo "To stop the service: sudo systemctl stop $SERVICE_NAME"
echo "To disable auto-start: sudo systemctl disable $SERVICE_NAME"
