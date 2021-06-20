#include <iostream>
#include <memory>
#include <chrono>
#include <string>
 
#include "cxxopts.hpp"
 

#include <torch/script.h>
#include <torch/torch.h>
//#include <c10/cuda/CUDAStream.h>
//#include <ATen/cuda/CUDAEvent.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

class Detector {
public:
    Detector(const std::string& model_path, const torch::DeviceType& device_type);
    Detector(const std::string& model_path);
    std::vector<std::vector<Detection>>
    Run(const cv::Mat& img, float conf_threshold, float iou_threshold);
    cv::Mat&
    draw_result(cv::Mat& img,
                const std::vector<std::vector<Detection>>& detections,
                const std::vector<std::string>& class_names);
    cv::Mat&
    pic2pic(cv::Mat& img,
            float conf_threshold, float iou_threshold,
            const std::vector<std::string>& class_names);
    cv::Mat&
    path2pic(const std::string img_path,
                       float conf_threshold, float iou_threshold,
                       const std::vector<std::string>& class_names);
private:
    static std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size = cv::Size(640, 640));

    static std::vector<std::vector<Detection>> PostProcessing(const torch::Tensor& detections,
                                                              float pad_w, float pad_h, float scale, const cv::Size& img_shape,
                                                              float conf_thres = 0.4, float iou_thres = 0.6);

    static void ScaleCoordinates(std::vector<Detection>& data, float pad_w, float pad_h,
                                 float scale, const cv::Size& img_shape);

    static torch::Tensor xywh2xyxy(const torch::Tensor& x);

    static void Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
                                 const at::TensorAccessor<float, 2>& det,
                                 std::vector<cv::Rect>& offset_box_vec,
                                 std::vector<float>& score_vec);
    torch::jit::script::Module module_;
    torch::Device device_;
    bool half_;
};


std::vector<std::string> LoadNames(const std::string& path) {
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline (infile,line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}


void Demo(cv::Mat& img,
        const std::vector<std::vector<Detection>>& detections,
        const std::vector<std::string>& class_names,
        bool label = true) {

    if (!detections.empty()) {
        for (const auto& detection : detections[0]) {
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

            if (label) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = class_names[class_idx] + " " + ss.str();

                auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline=0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                cv::rectangle(img,
                        cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                        cv::Point(box.tl().x + s_size.width, box.tl().y),
                        cv::Scalar(0, 0, 255), -1);
                cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                            font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
            }
        }
    }

    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Result", img);
    cv::waitKey(0);
}


int main(int argc, const char* argv[]) {
    cxxopts::Options parser(argv[0], "A LibTorch inference implementation of the yolov5");
    // TODO: add other args
    parser.allow_unrecognised_options().add_options()
            ("weights", "model.torchscript.pt path", cxxopts::value<std::string>())
            ("source", "source", cxxopts::value<std::string>())
            ("conf-thres", "object confidence threshold", cxxopts::value<float>()->default_value("0.25"))
            ("iou-thres", "IOU threshold for NMS", cxxopts::value<float>()->default_value("0.5"))
            ("gpu", "Enable cuda device or cpu", cxxopts::value<bool>()->default_value("false"))
            ("view-img", "display results", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print usage");

    auto opt = parser.parse(argc, argv);

    if (opt.count("help")) {
        std::cout << parser.help() << std::endl;
        exit(0);
    }


    // check if gpu flag is set
    bool is_gpu = opt["gpu"].as<bool>();

    // set device type - CPU/GPU
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && is_gpu) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    // load class names from dataset for visualization
    std::string classname;
    std::cout << "load classname path :";
    // std::cin >> classname;
    std::vector<std::string> class_names = LoadNames("/Users/cxu/Desktop/torch/weights/ship.name");
    if (class_names.empty()) {
        return -1;
    }

    // load network
    std::string weights = "/Users/cxu/Desktop/torch/weights/yolov5s.torchscript.pt";//opt["weights"].as<std::string>();
    auto detector = Detector(weights, device_type);

    // load input image
    std::string source = "/Users/cxu/Desktop/torch/images/bus_out.jpg" ;//opt["source"].as<std::string>();
    cv::Mat img = cv::imread(source);
    if (img.empty()) {
        std::cerr << "Error loading the image!\n";
        return -1;
    }

    // run once to warm up
    std::cout << "Run once on empty image" << std::endl;
    auto temp_img = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
    detector.Run(temp_img, 1.0f, 1.0f);

    // set up threshold
    float conf_thres = opt["conf-thres"].as<float>();
    float iou_thres = opt["iou-thres"].as<float>();

    // inference
    auto result = detector.Run(img, conf_thres, iou_thres);

    // visualize detections
    if (opt["view-img"].as<bool>()) {
        Demo(img, result, class_names);
    }

    cv::destroyAllWindows();
    return 0;
}
