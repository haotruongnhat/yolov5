pyinstaller --distpath pyinstaller\dist --workpath pyinstaller\build demthep_detect.spec

python export.py --weights epoch100_f23.pt --include onnx

python benchmark_onnx.py --data_dir D:\Projects\VSTech\yolov5\outputs\data_new_view\original_images --output_dir D:\Projects\VSTech\yolov5\outputs\data_new_view\output_more_image --weights data_new_view_july.onnx --save_overlay --save_label --print_score
python interfaces\split_datasets.py

python benchmark_onnx.py --data_dir D:\Projects\VSTech\yolov5\outputs\data_check_august\error --output_dir D:\Projects\VSTech\yolov5\outputs\data_check_august\error_out --weights data_new_view_july_more_image_3.onnx --save_overlay --save_label --print_score
python benchmark_onnx.py --data_dir D:\Projects\VSTech\yolov5\outputs\data_september --output_dir D:\Projects\VSTech\yolov5\outputs\data_september_out --weights data_new_view_july_more_image_3.onnx --save_overlay --save_label --print_score
python benchmark_onnx.py --data_dir D:\Projects\VSTech\yolov5\outputs\data_september --output_dir D:\Projects\VSTech\yolov5\outputs\data_september_out_error --weights data_sept_increase_augment_exp15.onnx --save_image --save_overlay --save_label --print_score


python benchmark_onnx.py --data_dir D:\Projects\VSTech\yolov5\outputs\data_september_checked --output_dir D:\Projects\VSTech\yolov5\outputs\data_september_out_error --weights bb3cd2e152a34391b2683b3bf3d6f4e6.onnx --save_image --save_overlay --save_label --print_score
