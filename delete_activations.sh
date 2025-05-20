#!/bin/bash

# Loop through the range from 40000 to 80939
for i in $(seq 40000 80939); do
    # Construct the filename
    filename="saved_activations_1b/activations_${i}.pt"
    
    # Check if the file exists
    if [ -f "$filename" ]; then
        # Delete the file
        rm "$filename"
        echo "Deleted: $filename"
    else
        echo "File not found: $filename"
    fi
done

echo "Deletion process completed."