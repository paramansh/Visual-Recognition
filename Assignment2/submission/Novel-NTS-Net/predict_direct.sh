rm output.txt
rm image_list.txt
ls test_images > image_list.txt
python3 test_predict.py --image_file=image_list.txt
cat output.txt
