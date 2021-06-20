#include "common.hpp"


ConvSiluBnImpl::ConvSiluBnImpl(int input_channel, int output_channel, int kernel_size, int stride) {
    cv1 = register_module("cv1", torch::nn::Conv2d(
        conv_options(input_channel,output_channel,kernel_size,stride)));
    bn = register_module("bn", torch::nn::BatchNorm2d(output_channel)); 

}
torch::Tensor ConvSiluBnImpl::forward(torch::Tensor x) {
    x = cv1->forward(x);
    x = bn(x);
    return torch::silu(x);
}

BottleneckImpl::BottleneckImpl(int c1, int c2, bool shortcut=true, int g=1, float e=0.5){
    int c_ = int(c2 * e);  // hidden channels
    cv1 = register_module("cv1", ConvSiluBn(c1, c_, 1, 1)); 
    cv2 = register_module("cv2", ConvSiluBn(c1, c_, 3, 1)); 
    add =   (shortcut and c1 == c2) ? 1 : 0 ;
}
torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
    torch::Tensor x1 = cv1->forward(x);
    x1 = cv2->forward(x1);
    if(add) return x1 + x;
    return x;
}


C3Impl::C3Impl(int c1, int c2, int n=1, bool shortcut=0, int g=1, float e=0.5){
    int c_ = int(c2 * e) ;  //hidden channels
    cv1 = ConvSiluBn(c1,c_,1,1);
    cv2 = ConvSiluBn(c1,c_,1,1);
    cv3 = ConvSiluBn(2* c_,c2,1,1);
    cv1 = register_module("cv1",cv1);
    cv2 = register_module("cv2",cv2);
    cv3 = register_module("cv3",cv3);
    for (int i =0;i++;i<n){
        m->push_back(Bottleneck(c_, c_, shortcut, g, e=1.0));
    }
    m = register_module("m",m);
}
torch::Tensor C3Impl::forward(torch::Tensor x) {
    torch::Tensor c1 = cv1->forward(x);
    c1 = m->forward(c1);
    x = cv2->forward(x);
    c10::ArrayRef<torch::Tensor> mix( vector<torch::Tensor>{c1,x} );
    return cv3->forward(torch::cat(mix, 1));
};

SPPImpl::SPPImpl(int c1, int c2, vector<int> k){
    int c_ = c2 / 2 ;  //hidden channels
    cv1 = ConvSiluBn(c1,c_,1,1);
    cv2 = ConvSiluBn(c_ * k.size(),c2,1,1);
    cv1 = register_module("cv1",cv1);
    cv2 = register_module("cv2",cv2); 
 
    for (int i =0;i++;i<k.size()){
        m->push_back(torch::nn::MaxPool2d(maxpool_options(k[i], 1, k[i]/2)));
    }
    m = register_module("m",m);
};
torch::Tensor SPPImpl::forward(torch::Tensor x) {
    torch::Tensor c1(cv1->forward(x));
    vector<torch::Tensor> temp{c1};
    for (torch::nn::MaxPool2d mm : m) temp.push_back(mm->forward(c1));
    c10::ArrayRef<torch::Tensor> mix( temp );
    return cv2->forward(torch::cat( mix , 1));
};

FocusImpl::FocusImpl(int c1, int c2, int k=1, int s=1, int p=0, int g=1, bool act=true){ 
    conv = ConvSiluBn(4*c1,c2,k,s,p,g); 
    conv = register_module("conv",conv);
};
torch::Tensor FocusImpl::forward(torch::Tensor x) {
    vector<torch::Tensor> temp;
    temp.push_back( x.index({"...", Slice({0, -1, 2}), Slice({0, -1, 2})})); //[x[..., ::2, ::2]
    temp.push_back( x.index({"...", Slice({1, -1, 2}), Slice({0, -1, 2})})); //[x[..., ::2, ::2]
    temp.push_back( x.index({"...", Slice({0, -1, 2}), Slice({1, -1, 2})})); //[x[..., ::2, ::2]
    temp.push_back( x.index({"...", Slice({1, -1, 2}), Slice({1, -1, 2})})); //[x[..., ::2, ::2]
    c10::ArrayRef<torch::Tensor> mix( temp );
    return conv->forward(torch::cat(mix,1));
}; 
 
ConcatImpl::ConcatImpl(int dim ):d(dim){};
torch::Tensor ConcatImpl::forward(vector<torch::Tensor> x) {
    c10::ArrayRef<torch::Tensor> mix( x );
    return conv->forward(torch::cat(mix,d));
}; 
