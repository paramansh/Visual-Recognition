rm output.txt
rm image_list.txt
python3 test.py --tf=test_images/
echo "Coarse classification Complete"
python3 test_predict.py --image_file=image_list.txt
python3 sort.py
