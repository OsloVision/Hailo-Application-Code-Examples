#!/bin/bash

# Install Hailo Instance Segmentation systemd service
SERVICE_NAME="hailo-instance-segmentation.service"
SERVICE_FILE="./hailo-instance-segmentation.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "Installing Hailo Instance Segmentation systemd service..."

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Service file $SERVICE_FILE not found!"
    exit 1
fi

# Copy service file to systemd directory (requires sudo)
sudo cp "$SERVICE_FILE" "$SYSTEMD_DIR/"

# Reload systemd daemon
sudo systemctl daemon-reload

echo "Service installed successfully!"
echo ""
echo "To manage the service, use these commands:"
echo "  Start:   sudo systemctl start $SERVICE_NAME"
echo "  Stop:    sudo systemctl stop $SERVICE_NAME"
echo "  Enable:  sudo systemctl enable $SERVICE_NAME  (start on boot)"
echo "  Disable: sudo systemctl disable $SERVICE_NAME (don't start on boot)"
echo "  Status:  sudo systemctl status $SERVICE_NAME"
echo "  Logs:    journalctl -u $SERVICE_NAME -f"
echo ""
echo "To enable and start the service now:"
echo "  sudo systemctl enable --now $SERVICE_NAME"
