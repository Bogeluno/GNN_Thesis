#!/bin/bash
# Training settings 
NODAYS=60
EPOCHS=300
# RUN
echo "Starting Zones:"
python 5run/NN_Zones5.py $NODAYS $EPOCHS
echo "Starting No Zones:"
python 5run/NN_No_Zones5.py $NODAYS $EPOCHS
echo "Starting Encoded Zones:"
python 5run/NN_Encoded_Zones5.py $NODAYS $EPOCHS
echo "Starting Weather"
python 5run/NN_Weather5.py $NODAYS $EPOCHS
echo "Starting All Encoded:"
python 5run/NN_All_Encoded5.py $NODAYS $EPOCHS
#echo "Starting All:"
#python 5run/NN_All5.py $NODAYS $EPOCHS