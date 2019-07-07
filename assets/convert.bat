python ./tools/keras_to_tensorflow.py ^
-input_model_file "../model/trained_model.h5" ^
-output_model_file "../model/eye_close_detector.pb"

copy "..\model\eye_close_detector.pb" "C:\Users\artye\Documents\Visual Studio Projects\Estimate_CPP\Estimate_CPP\assets\eye_close_detector.pb"
pause