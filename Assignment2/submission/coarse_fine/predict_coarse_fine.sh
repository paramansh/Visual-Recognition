#!/bin/sh
rm output.txt
rm -r coarse_output``
python3 test.py --tf=test_images/
echo "Fine grained Classification complete..."
echo " "
echo " "
echo " "
echo " "

python3 test_predict.py --imclass='aircrafts'
python3 test_predict.py --imclass='birds'
python3 test_predict.py --imclass='cars'
python3 test_predict.py --imclass='dogs'
python3 test_predict.py --imclass='flowers'

echo "Fine grained Classification complete..."
echo " "
echo " "
echo " "
echo " "

cat output.txt
