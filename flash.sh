#!/bin/bash
# ESP-CLAW Flash Script for ESP32S3
# Connect your ESP32S3 via USB, then run this script

PORT="${1:-/dev/ttyUSB0}"

echo "Flashing ESP-CLAW to $PORT..."
echo "WiFi: Analist | LLM: Ollama (192.168.7.12:11434) | Telegram: enabled"
echo ""

python3 -m esptool \
    --chip esp32s3 \
    --port "$PORT" \
    -b 460800 \
    --before default_reset \
    --after hard_reset \
    write_flash \
    --flash_mode dio \
    --flash_size 4MB \
    --flash_freq 80m \
    0x0       build/bootloader/bootloader.bin \
    0x8000    build/partition_table/partition-table.bin \
    0x10000   build/espclaw.bin

echo ""
echo "Done! ESP-CLAW is now running with Ollama + Telegram."
