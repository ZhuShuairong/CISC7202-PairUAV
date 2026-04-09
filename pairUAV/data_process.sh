#!/bin/bash

# Uses Data already extracted in AutoDL
AUTODL_TMP_DIR="/root/autodl-tmp/university"

echo "Creating training set from University-Release drone data..."
mkdir -p train_tour
cp -r ${AUTODL_TMP_DIR}/University-Release/University-Release/train/drone/* train_tour/

echo "Creating symlinks for PairUAV data..."
ln -sf ${AUTODL_TMP_DIR}/PairUAV/train ./train
ln -sf ${AUTODL_TMP_DIR}/PairUAV/test ./test
ln -sf ${AUTODL_TMP_DIR}/PairUAV/test_tour ./test_tour

echo "Data setup complete!"
