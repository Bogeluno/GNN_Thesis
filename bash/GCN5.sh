#!/bin/bash
# Training settings 
NODAYS=60
EPOCHS=400
# RUN
#echo "Starting No Zones:"
#python 5run/GCN_No_Zones5.py $NODAYS $EPOCHS
echo "Starting Encoded Zones:"
python 5run/GCN_Encoded_Zones5.py $NODAYS $EPOCHS
echo "Starting Weather"
python 5run/GCN_Weather5.py $NODAYS $EPOCHS
echo "Starting All Encoded:"
python 5run/GCN_All_Encoded5.py $NODAYS $EPOCHS
echo "Starting Zones:"
python 5run/GCN_Zones5.py $NODAYS $EPOCHS
echo "Starting All:"
python 5run/GCN_All5.py $NODAYS $EPOCHS

