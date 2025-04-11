#!/bin/bash

# Create the target directory if it doesn't exist
mkdir -p data/MultiAgentJPG

# Google Drive file ID extracted from the URL
FILE_ID="1CVqePMvmz8Zs_CCWIGu8LP7rF02H6_jQ"

# Temporary zip file location
TMP_ZIP="./temp_download.zip"

echo "Checking if gdown is installed..."
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. You can install it using:"
    echo "  pip install gdown"
    echo "Falling back to alternate download method..."
    
    # Try curl with cookies method
    echo "Using curl method instead..."
    curl -L -c ./cookie.txt "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
    curl -L -b ./cookie.txt "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie.txt)&id=${FILE_ID}" -o ${TMP_ZIP}
    rm -f ./cookie.txt
else
    echo "Downloading file using gdown..."
    gdown --id ${FILE_ID} -O ${TMP_ZIP}
fi

# Check if download was successful
if [ ! -f "${TMP_ZIP}" ]; then
    echo "Download failed. Please download the file manually from Google Drive."
    exit 1
fi

echo "Download complete. Extracting files..."

# Extract the zip file to the target directory
unzip -q ${TMP_ZIP} -d data/MultiAgentJPG

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Extraction failed. The downloaded file might be corrupted or not a zip file."
    rm -f ${TMP_ZIP}
    exit 1
fi

echo "Extraction complete. Cleaning up..."

# Remove the temporary zip file
rm -f ${TMP_ZIP}

echo "Done! Files have been extracted to data/MultiAgentJPG/"