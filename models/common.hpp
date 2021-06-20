#include <torch/script.h>
#include <torch/nn/module.h>
#include <torch/nn/modules.h>
#include <torch/torch.h>

#include <iostream>
#include <typeinfo>
using namespace torch;
using namespace std;

inline int autopad(int k) {return k/2 ;};

inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, int64_t groups = 1,bool with_bias = false) 
{
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    if(padding == 0){
        conv_options.padding(stride / 2);
    }else  conv_options.padding(padding);
    conv_options.groups(groups);
    conv_options.bias(with_bias);
    return conv_options;
}
inline torch::nn::MaxPool2dOptions maxpool_options(int kernel_size, int stride,int padding){
    torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
    maxpool_options.stride(stride);
    maxpool_options.padding(padding);
    return maxpool_options;
}


class ConvSiluBnImpl : public torch::nn::Module {
public:
    ConvSiluBnImpl(int input_channel=3, int output_channel=64, int kernel_size = 3, int stride = 1);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor fuseforward(torch::Tensor x);
private: // Declare layers
    torch::nn::Conv2d cv1{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr }; 
};
TORCH_MODULE(ConvSiluBn);


class BottleneckImpl: public torch::nn::Module{
public:
    BottleneckImpl(int c1, int c2, bool shortcut=true, int g=1, float e=0.5);
    torch::Tensor forward(torch::Tensor x);
private:
    ConvSiluBn cv1{ nullptr };
    ConvSiluBn cv2{ nullptr };
    bool add{ 0 };
};
TORCH_MODULE(Bottleneck);


class C3Impl: public torch::nn::Module{
public:
    C3Impl(int c1, int c2, int n=1, bool shortcut=true, int g=1, float e=0.5);
    torch::Tensor forward(torch::Tensor x);
private:
    ConvSiluBn cv1{ nullptr };
    ConvSiluBn cv2{ nullptr }; 
    ConvSiluBn cv3{ nullptr }; 
    torch::nn::Sequential m{ nullptr };
};
TORCH_MODULE(C3);


class SPPImpl: public torch::nn::Module{
public:
    SPPImpl(int c1, int c2, std::vector<int> k = std::vector<int>{5, 9, 13});
    torch::Tensor forward(torch::Tensor x);
private:
    ConvSiluBn cv1{ nullptr };
    ConvSiluBn cv2{ nullptr }; 
    torch::nn::ModuleList m{ nullptr };
};
TORCH_MODULE(SPP);


class FocusImpl: public torch::nn::Module{
public:
    FocusImpl(int c1, int c2, int k=1, int s=1, int p=0, int g=1, bool act=true);
    torch::Tensor forward(torch::Tensor x);
private:
    ConvSiluBn conv{ nullptr };
};
TORCH_MODULE(Focus); 


class ConcatImpl: public torch::nn::Module{
public:
    ConcatImpl(int dim);
    torch::Tensor forward(std::vector<torch::Tensor> x); 
private:
    int d{0};
};
TORCH_MODULE(Concat);  