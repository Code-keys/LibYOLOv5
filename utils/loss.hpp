#include <torch/torch.h>
#include <vector>
#include <string> 

using namespace std;


inline float smooth_BCE(float eps=0.1) {return {1.0 - 0.5 * eps, 0.5 * eps};};

class BCEBlurWithLogitsLoss: public torch::nn::Module{};

class FocalLoss: public torch::nn::Module{};

class QFocalLoss: public torch::nn::Module{};
 
bool compute_loss(vector<torch::Tensor> p, vector<vector<float> > targets, torch::nn::Module model);

bool build_targets(vector<torch::Tensor> p, vector<vector<float> > targets, torch::nn::Module model);