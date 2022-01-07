#!/bin/bash
# Training settings 
NODAYS=60
EPOCHS=400
# RUN
echo "Starting No Zones:"
python 5run/GAT_No_Zones5.py $NODAYS $EPOCHS
echo "Starting Encoded Zones:"
python 5run/GAT_Encoded_Zones5.py $NODAYS $EPOCHS
echo "Starting Weather"
python 5run/GAT_Weather5.py $NODAYS $EPOCHS
echo "Starting All Encoded:"
python 5run/GAT_All_Encoded5.py $NODAYS $EPOCHS
echo "Starting Zones:"
python 5run/GAT_Zones5.py $NODAYS $EPOCHS
echo "Starting All:"
python 5run/GAT_All5.py $NODAYS $EPOCHS