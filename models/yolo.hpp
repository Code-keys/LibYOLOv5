#include <iostream>
#include <string>
#include <vector>
#include <set>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "yaml.h"
#include "cxxopts.hpp"

#include "torch_utils.hpp"
#include "general.hpp" 
#include "common.hpp"

using namespace std;


class DetectImpl: public torch::nn::Module{
public:
    DetectImpl(int nc, torch::Tensor anchors, vector<int> ch);
    vector< torch::Tensor > forward();
    static torch::Tensor stride{NULL};//#/ strides computed during build
    static bool Export{0};//# onnx export
    int nc,no,nl,na;
    torch::Tensor a, anchor_grid; 
    torch::nn::ModuleList m{nullptr} ;
    vector< torch::Tensor > grid;
    bool training{0};
private:
    torch::Tensor _make_grid()  ;
}
TORCH_MODULE(Detect);
 

class ModelImpl: public torch::nn::Module{
public:
    ModelImpl(const string cfg_path, int ch=3, int nc=NULL );
    vector< torch::Tensor > forward(); 
    vector< torch::Tensor > forward_once(); 
    static void fuse();

    bool training{0};
    torch::Tensor stride;
    vector<string> names;
    torch::nn::Sequential model{nullptr};

private:
    YAML::Node yaml;
    std::set save;

    static bool _parse_model(auto d, int ch)  ;// model_dict, input_channels(3)
    static void _initialize_biases();;
    static void info();
    static void _print_biases();
    static void _print_weights();
}
TORCH_MODULE(Model);
