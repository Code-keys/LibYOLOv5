#include "__init__.h"
#include "cxxopts.hpp"
#include "torch/script.h"
#include "torch/torch.h"

using namespace std;

int main(){

    torch::Tensor a = torch::empty({2, 4});
    std::cout << a << std::endl;
    torch::Tensor a = torch::rand({2, 3});
    std::cout << a << std::endl;
    torch::Tensor b = torch::ones({2, 4});
    std::cout << b << std::endl;


}