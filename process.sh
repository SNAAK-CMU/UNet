#!/bin/bash

python3 make_single_channel.py

echo "MASKS ARE NOW SINGLE CHANNEL"

python3 voc_annotation.py

echo "TRAINING IS SET UP"

echo "STARTING TRAINING"

python3 train.py

echo "TRAINING COMPLETE"


