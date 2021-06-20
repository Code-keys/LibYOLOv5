#pragma once
#include "opencv2/opencv.hpp"


enum Det {
    tl_x = 0,
    tl_y = 1,
    br_x = 2,
    br_y = 3,
    score = 4,
    class_idx = 5
};

struct Detection {
    cv::Rect bbox;
    float score;
    int class_idx;
};

bool test_module(){
    YAML::Node config = YAML::LoadFile("../config.yaml");
    auto yolov5 = model(".yaml",3);
    auto yolov5name = yolov5->named_parameters();
    for (auto n = yolov5name.begin(); n != yolov5name.end(); n++)
    {
        std::cout<<(*n).key()<<std::endl;
    };
    return 0;
};