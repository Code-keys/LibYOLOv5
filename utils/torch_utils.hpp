#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

Torch::Tensor xyxy2xywh(Torch::Tensor x){
    // Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    Torch::Tensor y{x.clone()};
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  // x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  // y center
    y[:, 2] = x[:, 2] - x[:, 0]  // width
    y[:, 3] = x[:, 3] - x[:, 1]  // height
    return y;
}


Torch::Tensor xywh2xyxy(Torch::Tensor x){
    // Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    Torch::Tensor y{x.clone()};
    y[:, 0] = x[:, 0] - x[:, 2] / 2  // top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  // top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  // bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  // bottom right y
    return y;
}