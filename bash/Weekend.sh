#!/bin/bash
# Training settings 
NODAYS=60
EPOCHS=400
# RUN
echo "Starting GAT All:"
python 5run/GAT_All5.py $NODAYS $EPOCHS
echo "Starting GCN All:"
python 5run/GCN_All5.py $NODAYS $EPOCHS
echo "Starting GCN Zones:"
python 5run/GCN_Zones5.py $NODAYS $EPOCHS
echo "Starting NN All:"
python 5run/NN_All5.py $NODAYS $EPOCHS
echo "Starting NN Zones:"
python 5run/NN_Zones5.py $NODAYS $EPOCHS