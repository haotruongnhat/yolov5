pyinstaller --distpath pyinstaller\dist --workpath pyinstaller\build --icon=demthep.ico demthep_detect.spec

python export.py --weights epoch100_f23.pt --include onnx