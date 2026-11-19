# Assets and attribution

Footage: **The University of Auckland**, “2017 Interfaculty Touch Rugby Tournament”, published 2017. Licence record: https://commons.wikimedia.org/wiki/File:2017-Interfaculty-Touch-Rugby-Tournament_1080p-Nueva_Zelanda.webm (CC BY 2.5, https://creativecommons.org/licenses/by/2.5/).
Original source listed by Commons: https://www.youtube.com/watch?v=CXWyq9hcZTg . Retrieved from that original source in video-only 360p format after Commons returned HTTP 429. The local 640x360 copy therefore has a different checksum from the Commons original. Changes: selected time interval, omitted audio, bounding boxes and text overlays. Source credit appears on the output video and report. No endorsement or affiliation is implied.

Model: OpenCV Zoo **YOLOX-S**, `models/object_detection_yolox/object_detection_yolox_2022nov.onnx`, upstream revision `48b10fa505aeb51fbd127fc60a5c660040e823b7`. Model-folder licence: Apache-2.0, reproduced in `assets/LICENSE-YOLOX.txt`. The downloaded model SHA256 was checked against the upstream Git LFS pointer: `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.
Model source: https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox . Pre/postprocessing uses the format documented by that model and OpenCV: https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html . The small detector wrapper follows the upstream coordinate-decoding formula; the tracker and evaluator are independently implemented for this demo.

This package does not train a model, export a new ONNX model, identify people, infer physical speed or assess athletic performance. It is a local portfolio artifact; no media or code has been published or sent to Athlaite.
