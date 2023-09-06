#!/bin/bash

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_PATH="$CURR_DIR/pento-parser-poc/test/data/images/IMG_2348.png"

echo "$IMAGE_PATH"
echo "image=@/${IMAGE_PATH}"

curl --verbose -X POST -H "Content-Type: multipart/form-data" -F "image=@${IMAGE_PATH}" localhost:5000/parse-solution
