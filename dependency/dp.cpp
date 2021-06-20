#include "dp.h"
#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <assert.h>
#include <torch/torch.h>
#include <torch/script.h>

#include "general.hpp"


class C3Impl :public torch::nn::Module{

};
TORCH_MODULE(C3);

int main(){

    YAML::Node a;
    a = YAML::LoadFile("/Users/cxu/Desktop/libyolov5/cfg/yolov5s.yaml");

    std::cout << a["backbone"][9][3][1].as<std::string>()  << std::endl; 
    for(auto i:a["backbone"][9][3]){
        std::cout << i.as<std::string>()  << std::endl; 
    }
    std::vector<float> scales;
    torch::Tensor scales_ = torch::tensor(scales);
    std::string s{"1"};
    s == "1" ?printf("6"):0; 
    int aa = cx::trans2num<int>(s);
    std::cout << aa << std::endl;  
    char c;
    cx::make_divisible(12.5);

    C3 lrelu;
    std::cout << lrelu->name();
    int i;
    for(;;){
        int j=0;
        i++;
        if(i > 1000000) break ;
    }
    assert(0 >= 9);

}