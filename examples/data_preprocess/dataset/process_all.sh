#!/bin/bash

# Run process_router.py
echo "Starting process_router.py..."
python data_router/process_router.py

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "process_router.py completed successfully."
else
    echo "Error: process_router.py failed. Exiting."
    exit 1
fi

# Run process_rec.py
echo "Starting process_rec.py..."
python data_rec/process_rec.py

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "process_rec.py completed successfully."
else
    echo "Error: process_rec.py failed. Exiting."
    exit 1
fi

# Run process_passage.py
echo "Starting process_passage.py..."
python data_ms_marco/process_passage.py

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "process_passage.py completed successfully."
else
    echo "Error: process_passage.py failed. Exiting."
    exit 1
fi

# Run combine_direct_data.py
echo "Starting combine_direct_data.py..."
python combine_direct_data.py

# Check if the previous command was successful

if [ $? -eq 0 ]; then
    echo "combine_direct_data.py completed successfully."
else
    echo "Error: combine_direct_data.py failed. Exiting."
    exit 1
fi

# Run combine_iterative_data.py
echo "Starting combine_iterative_data.py..."
python combine_iterative_data.py

# Check if the previous command was successful

if [ $? -eq 0 ]; then
    echo "combine_iterative_data.py completed successfully."
else
    echo "Error: combine_iterative_data.py failed. Exiting."
    exit 1
fi

echo "All scripts have finished running."

