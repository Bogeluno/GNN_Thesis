#!/bin/bash
# Training settings 
NODAYS=60
EPOCHS=300
# RUN
echo "Starting No Zones:"
python GAT_No_Zones.py $NODAYS $EPOCHS
echo "Starting Zones:"
python GAT_Zones.py $NODAYS $EPOCHS
echo "Starting Encoded Zones:"
python GAT_Encoded_Zones.py $NODAYS $EPOCHS
echo "Starting Weather"
python GAT_Weather.py $NODAYS $EPOCHS
echo "Starting All:"
python GAT_All.py $NODAYS $EPOCHS
echo "Starting All Encoded:"
python GAT_All_Encoded.py $NODAYS $EPOCHS