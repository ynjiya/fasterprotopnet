#!/bin/bash

# Directory path
DIR="datasets/CUB_200_2011"

# Check if directory exists
if [ -d "$DIR" ]; then
    echo "Directory $DIR already exists."
else
    echo "Directory $DIR does not exist. Creating now..."
    mkdir -p "$DIR"
    echo "Directory $DIR created."
fi

# Cropping the dataset
if [ -d "datasets/cub200_cropped" ]; then
    echo "Directory datasets/cub200_cropped already exists."
else
    echo "Cropping the dataset..."
    python3 crop_dataset.py "${DIR}/images" "${DIR}/images.txt" "${DIR}/bounding_boxes.txt" "datasets/cub200_cropped/tmp"
    if [ $? -eq 0 ]; then
        echo "Dataset cropped successfully."
    else
        echo "Error: Dataset cropping failed."
        exit 1
    fi
fi


# Splitting the dataset
echo "Splitting the dataset..."
python3 split_dataset.py "${DIR}/images.txt" "${DIR}/train_test_split.txt" "datasets/cub200_cropped/tmp" "datasets/cub200_cropped"
if [ $? -eq 0 ]; then
    echo "Dataset splitted successfully."
else
    echo "Error: Dataset split failed."
    exit 1
fi