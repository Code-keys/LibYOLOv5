#include "yolo.hpp"


DetectImpl::DetectImpl( int nc, torch::Tensor anchors, std::vector<int> ch):
                        nc(nc),no(nc+5),nl(anchors.size()),na((anchors[0].size())){
    a = torch::Tensor(anchors).float().view({nl, -1, 2});
    grid = std::vector<torch::Tensor>(torch::zeros({1}), nl) ;
    training = 0;
    this.register_buffer("anchors",a); //shape(nl,na,2)
    this.register_buffer("anchor_grid",a.clone().view({nl, 1, -1, 1, 1, 2}));  ////# shape(nl,1,na,1,1,2)

    for (int i: ch){
        m->push_back(torch::nn::Conv2d( conv_options(i,no * na,1,1) )); // output conv
    }
    this.register_module("m",m);
}
std::vector<torch::Tensor> DetectImpl::forward(std::vector<torch::Tensor>& x){
    std::vector<torch::Tensor> z{NULL} ; //inference output 
    torch::Tensor y;
    training = this.is_training() | Export;
    for (int i=0; i < nl; i++ ){
        x[i] = m[i]->forward(x[i]);
        int bs{x[i].size(0)}, ny{x[i].size(2)}, nx{x[i].size(3)}; 
        x[i] = x[i].view({bs, na, no, ny, nx}).permute({0, 1, 3, 4, 2}).contiguous();
        if (!training){
                // if grid[i].shape[2:4] != x[i].shape[2:4]:
            grid[i] = _make_grid(nx, ny).to(x[i].device_);
            y = x[i].sigmoid();
            y.index({ "...",Slice({0,2})}) = y.index({ "...",Slice({0,2})})* 2.0- 0.5 +
                    grid[i].to(x[i].device_)) * this.stride[i];
            y.index({ "...",Slice({2,4})}) = (y.index({ "...",Slice({2,4})})* 2.0)**2*anchor_grid[i];  // wh
            z.push_back(y.view({bs, -1, no}));
        }
    }
    return  training ? x : {torch::cat(z,1), x};
};
torch::Tensor DetectImpl::_make_grid(int nx=20, int ny=20) //const
{
    int yv, xv ;
    std::vector<torch::Tensor> mesh = torch::meshgrid({torch::arange(ny), torch::arange(nx)}); 
    return torch::stack({mesh[0], mesh[1]}, 2).view({1, 1, ny, nx, 2}).float();
}


ModelImpl::ModelImpl(const string cfg_path, int c1=3, int nc=NULL ){
    yaml = YAML::LoadFile(cfg_path);
    bool training{0};
    if (nc && nc != yaml['nc'].as<int> ):
        printf("Overriding model.yaml nc=%d with nc=%d" %(yaml['nc'], nc))
        yaml['nc'] = nc  # override yaml value
    _parse_model(yaml, c1);  // model, savelist 
    for(int i=0;i<nc;i++) names.push_back('0'+ i);
    torch::nn::Module _m = model[-1];
    if ( _m.name() == "Detect" ){
        model[-1].stride = stride;
        model[-1].anchors /= stride.view({-1, 1, 1});
    }

    info()；
}
std::vector< torch::Tensor > ModelImpl::forward();{
}
void ModelImpl::info(){
}
void ModelImpl::fuse(){ 
    printf('Fusing layers... \n')
    printf('oh my mistack, not implement ! \n')
    // for (m : model.modules()):
    //     if (m.name() == "Conv" and hasattr(m, 'bn')):
    //         m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
    //         delattr(m, 'bn')  # remove batchnorm
    //         m.forward = m.fuseforward  # update forward
    self.info()
}

bool ModelImpl::_parse_model(auto& const d, int c1 = 3){
    // stride    model    save
    printf('\n%3s%18s%3s%10s  %-40s%-30s\n' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    int nc{d['nc'].as<int>()};
    float gd{d['depth_multiple'].as<float>()}, gw{d['width_multiple'].as<float>()}; 
    std::vector<int> _a;int row{0},col{0};
    for(int i=0;i < d['anchors'].size();i++ ){ row++;
        for(int j=0;j<i.size(),j++) _a.push_back(d['anchors'][i][j].as<int>());
    } //torch::Tensor anchors = torch::tensor(_a).view({row,int(_a.size()/row)});
    torch::Tensor anchors = torch::from_blob(_a.data(), {row,row,_a.size()/row});
    na = col / 2; no = na * (5 + nc);
    
    set<int> save; std::vector<int> ch(c1,1);
    torch::nn::Sequential layers;  this.model
    int c2{c1};
    std::vector<NodeType::Sequence> _layer; //std::vector<string> args;
    for (NodeType::Sequence i : d['backbone']) _layer.push_back(i);
    for (NodeType::Sequence i : d['head']) _layer.push_back(i);
    for (   int i = 0 :  _layer.size(),  i++ ){

        int n{max(int(_layer[i][1].as<int>() * gd ),1)}; 
        std::string m{_layer[i][2].as<string>()}; // args.clear();  for(auto arg:_layer[i][3]) args.push_back(arg.as<std::string>()) ; 
        layers->reset();    c2=0;
        if(m=="Conv"|| m=="Bottleneck" || m=="SPP"|| m== "Focus" || m=="C3" || m== "DWConv" || m=="MixConv2d"|| m== "CrossConv" || m== "BottleneckCSP" || "nn.BatchNorm2d" )
        {
            int f{_layer[i][0].as<int>()};  f = f ? f+1 : ch[1+i+f];
            c1 = ch[f];     save.insert(f);
            c2 = _layer[i][3][0].as<int>();
            c2 = cx::make_divisible(c2 * gw);   // if c2 != no else c2
            for(int ct = 0; ct < n; ct++){
                if(m=="C3") layers->push_back(C3(c1, c2, n, _layer[i][3].size() > 1 )); 
                if(m=="SPP"){
                    std::vector<int> kernel;
                    for(auto i : _layer[i][3][1]) kernel.push_back(i.as<int>());
                    layers->push_back(SPP(c1, c2, kernel));
                } 
                if(m=="Focus") layers->push_back(Focus(c1, c2, _layer[i][3][1].as<int>()));
                if(m=="DWConv") layers->push_back(DWConv(c1, c2));
                if(m=="MixConv2d") layers->push_back(MixConv2d(c1, c2));
                if(m=="CrossConv") layers->push_back(CrossConv(c1, c2, _layer[i][3][1].as<int>(), _layer[i][3][2].as<int>()));
                if(m=="Bottleneck") layers->push_back(Bottleneck(c1, c2, _layer[i][3].size() > 1));
                if(m=="BottleneckCSP") layers->push_back(BottleneckCSP(c1, c2, n, _layer[i][3].size() > 1));
                }
        }else if (m=="Concat")
        {
            vector<int> f, c;    int _f{i.as<int>()};
            for(auto i:_layer[i][0])  {
                _f =  _f > 0 ? _f : _f + 1 + i; 
                f.push_back(1+_f);
                c.push_back(ch[1+i+_f]);
                c2 += ch[1+i+_f];
            }
            layers->push_back(Concat(c, c2, _layer[i][3][0].as<int>()));
        }else if (m == "nn.BatchNorm2d")
        {   int f{_layer[i][0].as<int>()}; if(f < 0) f = ch[1+i+f];
            layers->push_back(torch::nn::BatchNorm2d(c2));
        }else if (m == "Detect")
        {   vector<int> f;
            for(auto i:_layer[i][0]){ f.push_back(ch[1+  i.as<int>()]); } ;//_layer[0].type() == 3
            layers->push_back(Detect(nc, anchors, f));
        }else if (m == "Contract")
        {
        }else if (m == "CenterHead")
        {        
        }else if (m == "PBMudle")
        {      
        }else if (m == "CBAM")
        {
        }else assert( "layer not support !" == "" );

        ch.push_back(c2);
        model->push_bach(layers);

        int np {0};  for(auto x : layers) np += x->parameters().numel() ;// number params
        string t = layers[0]->name();

        printf('%3s %18s %3s %10.0f  %-40  [ %4d ] ' % (i, "from", n, np, t, c2));
    }
    return 1;

}