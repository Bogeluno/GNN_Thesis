#!/bin/bash
# Training settings 
NODAYS=60
EPOCHS=300
# RUN
echo "Starting Zones:"
python NN_Zones.py $NODAYS $EPOCHS
echo "Starting All:"
python NN_All.py $NODAYS $EPOCHS
echo "Starting No Zones:"
python NN_No_Zones.py $NODAYS $EPOCHS
echo "Starting Encoded Zones:"
python NN_Encoded_Zones.py $NODAYS $EPOCHS
echo "Starting Weather"
python NN_Weather.py $NODAYS $EPOCHS
echo "Starting All Encoded:"
python NN_All_Encoded.py $NODAYS $EPOCHS