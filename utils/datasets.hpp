#include <iostream>
#include <string>
#include <torch/torch.h>
#include <torch/all.h>

using namespace std;
//遍历该目录下的.jpg图片
void load_data_from_folder(std::string image_dir, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label);

void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label)
{
    long long hFile = 0; //句柄
    struct _finddata_t fileInfo;
    std::string pathName;
    if ((hFile = _findfirst(pathName.assign(path).append("\\*.*").c_str(), &fileInfo)) == -1)
    {
        return;
    }
    do
    {
        const char* s = fileInfo.name;
        const char* t = type.data();

        if (fileInfo.attrib&_A_SUBDIR) //是子文件夹
        {
            //遍历子文件夹中的文件(夹)
            if (strcmp(s, ".") == 0 || strcmp(s, "..") == 0) //子文件夹目录是.或者..
                continue;
            std::string sub_path = path + "\\" + fileInfo.name;
            label++;
            load_data_from_folder(sub_path, type, list_images, list_labels, label);

        }
        else //判断是不是后缀为type文件
        {
            if (strstr(s, t))
            {
                std::string image_path = path + "\\" + fileInfo.name;
                list_images.push_back(image_path);
                list_labels.push_back(label);
            }
        }
    } while (_findnext(hFile, &fileInfo) == 0);
    return;
}


class dataSetClc:public torch::data::Dataset<dataSetClc>{
public:
    int class_index = 0;
    dataSetClc(std::string image_dir, std::string type){
        load_data_from_folder(image_dir, std::string(type), image_paths, labels, class_index-1);
    }
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override{
        std::string image_path = image_paths.at(index);
        cv::Mat image = cv::imread(image_path);
        cv::resize(image, image, cv::Size(224, 224)); //尺寸统一，用于张量stack，否则不能使用stack
        int label = labels.at(index);
        torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
        torch::Tensor label_tensor = torch::full({ 1 }, label);
        return {img_tensor.clone(), label_tensor.clone()};
    }
    // Override size() function, return the length of data
    torch::optional<size_t> size() const override {
        return image_paths.size();
    };
private:
    std::vector<std::string> image_paths;
    std::vector<int> labels;
};




int batch_size = 2;
std::string image_dir = "your path to\\hymenoptera_data\\train";
auto mdataset = myDataset(image_dir,".jpg").map(torch::data::transforms::Stack<>());
auto mdataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(mdataset), batch_size);
for(auto &batch: *mdataloader){
    auto data = batch.data;
    auto target = batch.target;
    std::cout<<data.sizes()<<target;
}