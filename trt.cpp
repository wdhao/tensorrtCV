#include "trt.h"
#include "utils.h"

trt::trt(const string &jsonPath)
{
    //read the .json file
    Json::Reader m_Reader;
    Json::Value root;
    ifstream fp;

    fp.open(jsonPath,ios::binary);
    m_Reader.parse(fp,root);
    param.input_c = root["input_c"].asInt();
    param.input_h = root["input_h"].asInt();
    param.input_w = root["input_w"].asInt();

    param.fp16 = root["fp16"].asBool();
    param.int8 = root["int8"].asBool();
    param.Div_255 = root["div_255"].asBool();
    param.cali_txt = root["cali_txt"].asString();
    param.cali_table = root["cali_table"].asString();

    param.ENGPath = root["ENGPath"].asString();
    param.weightPath = root["weightsDir"].asString();
    param.onnxPath = root["onnxPath"].asString();

    param.mean.clear();
    param.std.clear();
    for (int i = 0; i < param.input_c; i++){
        param.mean.push_back(root["Mean"][i].asFloat());
        param.std.push_back(root["Std"][i].asFloat());
    }
    param.inputBlobName = root["inputBlobName"].asString();
    param.outputBlobName = root["outputBlobName"].asString();
    param.maxBatchsize = root["maxBatchsize"].asInt();
    param.outputSize = root["outputSize"].asInt();
    param.layers = root["network"];

    fp.close();
}
trt::~trt()
{
    if(m_context){
        m_context->destroy();
        m_context = nullptr;
    }
    if(m_engine){
        m_engine->destroy();
        m_engine = nullptr;
    }
    if(m_Network){
        m_Network->destroy();
        m_Network = nullptr;
    }
    for(auto bindings : m_bindings){
        cudaFree(bindings);
    }
    free(m_cudaStream);
}
void trt::debug_print(ITensor *input_tensor,const string &head)
{
    cout << head<< " : ";

       for (int i = 0; i < input_tensor->getDimensions().nbDims; i++)
       {
           cout << input_tensor->getDimensions().d[i] << " ";
       }
       cout<<endl;
}
void trt::printWeight(Weights wts, int wtsSize)
{
    for(int i = 0;i<wtsSize;i++)
    {
        cout<<*((float*)wts.values + i)<<" ";
        if(i%100>99)
            cout<<endl;
    }
    cout<<endl;
}
vector<float> trt::loadWeights(const string &filePath)
{
    int size = 0;
    ifstream file(filePath,ios_base::binary);
    file.read((char*)&size,4);
    char *floatWeight = new char[size*4];
    float *fp = (float*)floatWeight;
    file.read(floatWeight,4*size);
    vector<float> weights(fp,fp+size);
    delete [] floatWeight;
    file.close();
    return weights;
}
void trt::onnx2trt()
{
    IBuilder *builder = createInferBuilder(m_logger);
    m_Network = builder->createNetworkV2(1U);
    IBuilderConfig* config = builder->createBuilderConfig();
    auto parser = nvonnxparser::createParser(*m_Network,m_logger);
    if (!parser->parseFromFile((param.onnxPath).c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "ERROR: could not parse the model.\n";
        exit(1);
    }
    builder->setMaxBatchSize(param.maxBatchsize);
    config->setMaxWorkspaceSize(1<<30);
    if(param.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    cout<<"Building the Engine..."<<endl;
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*m_Network,*config);
    assert(engine != nullptr);
    cout<<"Building complete!"<<endl;
    cout<<"Serializing the Engine..."<<endl;
    nvinfer1::IHostMemory *modelStream = engine->serialize();
    assert(modelStream != nullptr);
    ofstream outFile;
    outFile.open(param.ENGPath,ios::binary);
    outFile.write(static_cast<const char*>(modelStream->data()),modelStream->size());
    outFile.close();
    m_Network->destroy();
    engine->destroy();
    builder->destroy();
    modelStream->destroy();
    cout<<"create Engine success!"<<endl;
}
void trt::createENG()
{

    IBuilder *builder = createInferBuilder(m_logger);
    m_Network = builder->createNetworkV2(0U);
    IBuilderConfig* config = builder->createBuilderConfig();
    ITensor *input = m_Network->addInput(param.inputBlobName.c_str(),DataType::kFLOAT,Dims3{param.input_c,param.input_h,param.input_w});
    assert(input);
    Layers[param.inputBlobName] = input;
    unsigned int layerSize = param.layers.size();
    cout<<"This networt's layer number is : "<<layerSize<<endl;

    for (unsigned int var = 0; var < layerSize; ++var) {
        Json::Value layer;
        layer = param.layers[var];
        addLayer(layer);
    }

    builder->setMaxBatchSize(param.maxBatchsize);
    config->setMaxWorkspaceSize(1 << 30);
    if(param.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (param.int8){
        const string caliTxt = param.cali_txt;
        const string int8cali_table = param.cali_table;
        calibrator *m_calibrator = new calibrator(1,caliTxt,int8cali_table,param.input_c,
                                                  param.input_h,param.input_w,param.inputBlobName,
                                                  param.mean,param.std,param.Div_255);
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(m_calibrator);
    }
    cout<<"Building the Engine..."<<endl;
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*m_Network,*config);
    assert(engine != nullptr);
    cout<<"Building complete!"<<endl;
    cout<<"Serializing the Engine..."<<endl;
    nvinfer1::IHostMemory *modelStream = engine->serialize();
    assert(modelStream != nullptr);
    ofstream outFile;
    outFile.open(param.ENGPath,ios::binary);
    outFile.write(static_cast<const char*>(modelStream->data()),modelStream->size());
    outFile.close();
    m_Network->destroy();
    engine->destroy();
    builder->destroy();
    modelStream->destroy();
    cout<<"create Engine success!"<<endl;
}
void trt::addLayer(Json::Value layer)
{

    string layerStyle = layer["layerStyle"].asString();
    if (layerStyle == "conv")
        trt_conv(layer);
    else if(layerStyle == "deconv")
        trt_deconv(layer);
    else if(layerStyle == "padding")
        trt_padding(layer);
    else if(layerStyle == "preInput")
        trt_preInput(layer);
    else if (layerStyle == "focus")
        trt_focus(layer);
    else if (layerStyle == "bn")
        trt_bn(layer);
    else if(layerStyle == "active")
        trt_active(layer);
    else if (layerStyle == "pool")
        trt_pool(layer);
    else if (layerStyle == "Pool")
        trt_Pool(layer);
    else if (layerStyle == "fc")
        trt_fc(layer);
    else if (layerStyle == "eltwise")
        trt_elt(layer);
    else if (layerStyle == "concat")
        trt_concat(layer);
    else if (layerStyle == "slice")
        trt_slice(layer);
    else if (layerStyle == "softmax")
        trt_softmax(layer);
    else if (layerStyle == "shuffle")
        trt_shuffle(layer);
    else if (layerStyle == "matmul")
        trt_matmul(layer);
    else if (layerStyle == "topk")
        trt_topk(layer);
    else if (layerStyle == "reduce")
        trt_reduce(layer);
    else if (layerStyle == "constant")
        trt_constant(layer);
    else if(layerStyle == "cba")
        trt_convBnActive(layer);
    else if(layerStyle == "resnet")
        trt_resnetLayer(layer);
    else if (layerStyle == "resnet3")
        trt_resnet3(layer);
    else if(layerStyle == "upsample")
        trt_UpSample(layer);
    else if(layerStyle == "upsample_plugin")
        trt_UpSample_plugin(layer);
    else if(layerStyle == "GN")
        trt_groupNorm(layer);
    else if(layerStyle == "unary")
        trt_unary(layer);
    else if (layerStyle == "C3") {
        yolo_C3(layer);
    }
    else if(layerStyle == "yolo")
        trt_yolo(layer);
    else if(layerStyle == "spp")
        yolo_spp(layer);
    else {
        cout<<"no this layer style : "<<layerStyle<<endl;
        abort();
    }
}
void trt::inference_init(int batchsize)
{
    ifstream cache(param.ENGPath,ios::binary);
    cache.seekg(0,ios::end);
    const int engSize = cache.tellg();
    cache.seekg(0,ios::beg);
    void *modelMem = malloc(engSize);
    cache.read((char*)modelMem,engSize);
    cache.close();
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(m_logger);
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(modelMem,engSize);
    runtime->destroy();
    free(modelMem);
    if(! engine){
        cout<<"deserialize eng error!"<<endl;
        return;
    }
    m_engine = engine;
    m_context = m_engine->createExecutionContext();
    if(cudaStreamCreate(&m_cudaStream)!=0) return;
    int bindings = m_engine->getNbBindings();
    m_bindings.resize(bindings,nullptr);

    //cout<<param.inputBlobName<<"  ~~~~~~~~~~~~~~~~~~~"<<endl;
    inputIndex = m_engine->getBindingIndex(param.inputBlobName.c_str());
    //
    //cout<<param.outputSize<<endl;
    int flag = cudaMalloc(&m_bindings.at(inputIndex),batchsize * param.input_c * param.input_h * param.input_w * 4);
    if(flag != 0){
        cout<<"input malloc error!"<<endl;
        assert(0);
    }

    outputIndex = m_engine->getBindingIndex(param.outputBlobName.c_str());
    flag = cudaMalloc(&m_bindings.at(outputIndex),batchsize * param.outputSize * 4);
    if(flag != 0)
    {
        cout<<"output malloc error!"<<endl;
        assert(0);
    }

}
void trt::doInference(const float *input, int batchsize, float *output)
{
    CUDA_CHECK(cudaMemcpyAsync(m_bindings.at(inputIndex),input,batchsize * param.input_c * param.input_h * param.input_w * 4,
                           cudaMemcpyHostToDevice,m_cudaStream));

    m_context->enqueue(batchsize,m_bindings.data(),m_cudaStream,nullptr);

    CUDA_CHECK(cudaMemcpyAsync(output,m_bindings.at(outputIndex),batchsize * param.outputSize * 4,
                           cudaMemcpyDeviceToHost,m_cudaStream));

    cudaStreamSynchronize(m_cudaStream);
}
void trt::doInference_int(const float *input, int batchsize, int *output)
{
    CUDA_CHECK(cudaMemcpyAsync(m_bindings.at(inputIndex),input,batchsize * param.input_c * param.input_h * param.input_w * 4,
                           cudaMemcpyHostToDevice,m_cudaStream));

    m_context->enqueue(batchsize,m_bindings.data(),m_cudaStream,nullptr);

    CUDA_CHECK(cudaMemcpyAsync(output,m_bindings.at(outputIndex),batchsize * param.outputSize * 4,
                           cudaMemcpyDeviceToHost,m_cudaStream));

    cudaStreamSynchronize(m_cudaStream);
}
ITensor* trt::trt_convNet(ITensor* input, string weightFile, string biasFile,
                            int output_c, DimsHW kernel, DimsHW stride, DimsHW padding, DimsHW dilations,
                          int groups, bool pre, bool post)
{
    vector<float> weights;
    vector<float> bias;
    weights   = loadWeights(weightFile);
    bias = loadWeights(biasFile);
    int size = weights.size();
    if(size == 0){
        cout<<"load weights error! Please check "<<weightFile<<endl;
        assert(0);
    }
    Weights convWeights {DataType::kFLOAT,nullptr,size};
    Weights convBias {DataType::kFLOAT,nullptr,output_c};
    float *val_wt = new float[size];
    for (int i = 0;i < size; i++) {
        val_wt[i] = weights[i];
    }
    convWeights.values = val_wt;
    float *val_bias = new float[output_c];
    for (int i = 0;i < output_c; i++) {
        val_bias[i] = 0.0;
        if(bias.size() != 0){
            val_bias[i] = bias[i];
        }
    }
    convBias.values = val_bias;
    IConvolutionLayer *conv = m_Network->addConvolutionNd(*input,output_c,kernel,convWeights,convBias);
    conv->setStrideNd(stride);
    if (pre)
    {
        conv->setPrePadding(padding);
    }
    else if(post)
    {
        conv->setPostPadding(padding);
    }
    else
    {
        conv->setPaddingNd(padding);
    }
    conv->setDilationNd(dilations);
    conv->setNbGroups(groups);
    assert(conv);
    weights.clear();
    bias.clear();

    return conv->getOutput(0);
}
ITensor* trt::trt_deconvNet(ITensor* input, string weightFile, string biasFile,
                            int output_c, DimsHW kernel, DimsHW stride, DimsHW padding, DimsHW dilations,
                          int groups, bool pre, bool post)
{
    vector<float> weights;
    vector<float> bias;
    weights   = loadWeights(weightFile);
    bias = loadWeights(biasFile);
    int size = weights.size();
    if(size == 0){
        cout<<"load weights error! Please check "<<weightFile<<endl;
        assert(0);
    }
    Weights convWeights {DataType::kFLOAT,nullptr,size};
    Weights convBias {DataType::kFLOAT,nullptr,output_c};
    float *val_wt = new float[size];
    for (int i = 0;i < size; i++) {
        val_wt[i] = weights[i];
    }
    convWeights.values = val_wt;
    float *val_bias = new float[output_c];
    for (int i = 0;i < output_c; i++) {
        val_bias[i] = 0.0;
        if(bias.size() != 0){
            val_bias[i] = bias[i];
        }
    }
    convBias.values = val_bias;
    IDeconvolutionLayer *deconv = m_Network->addDeconvolutionNd(*input,output_c,kernel,convWeights,convBias);
    deconv->setStrideNd(stride);
    if (pre)
    {
        deconv->setPrePadding(padding);
    }
    else if(post)
    {
        deconv->setPostPadding(padding);
    }
    else
    {
        deconv->setPaddingNd(padding);
    }
    //deconv->setDilationNd(dilations);
    deconv->setNbGroups(groups);
    assert(deconv);
    weights.clear();
    bias.clear();

    return deconv->getOutput(0);
}
ITensor* trt::trt_bnNet(ITensor* input, string weightsPath,float eps)
{
    vector<float> weights;
    vector<float> bias;
    vector<float> mean;
    vector<float> var;
    string weightsFile = weightsPath + ".weight.wgt";
    string biasFile = weightsPath + ".bias.wgt";
    string meanFile = weightsPath + ".running_mean.wgt";
    string varFile = weightsPath + ".running_var.wgt";
    weights = loadWeights(weightsFile);
    bias = loadWeights(biasFile);
    mean = loadWeights(meanFile);
    var = loadWeights(varFile);
    unsigned int size = bias.size();
    if(weights.size()==0 || size ==0 || mean.size() ==0 || var.size() ==0)
    {
        cout<<"load weights error! Please check it!"<<weightsFile<<endl;
        assert(0);
    }
    Weights scale{DataType::kFLOAT,nullptr,size};
    Weights shift{DataType::kFLOAT,nullptr,size};
    Weights power{DataType::kFLOAT,nullptr,size};
    vector<float>bn_var;
    for (int i = 0; i < size; i++) {
        bn_var.push_back( sqrt(var.at(i)+eps));
    }
    float *shiftWt = new float[size];
    for (int i = 0; i < size; i++) {
        shiftWt[i] = bias.at(i)-(mean.at(i)*weights.at(i)/bn_var.at(i));
    }
    shift.values = shiftWt;
    float *scaleWt = new float[size];
    float *powerWt = new float[size];
    for (int i = 0; i <size; i++) {
        scaleWt[i] = weights.at(i)/bn_var.at(i);
        powerWt[i] = 1.0;
    }
    scale.values = scaleWt;
    power.values = powerWt;
    ScaleMode scaleMode = ScaleMode::kCHANNEL;
    IScaleLayer *batchNorm = m_Network->addScale(*input,scaleMode,shift,scale,power);
    assert(batchNorm);
    weights.clear();
    bias.clear();
    mean.clear();
    var.clear();
    bn_var.clear();

    return batchNorm->getOutput(0);
}
ITensor* trt::trt_activeNet(ITensor* input, string acti_type,float alpha,float beta)
{
    ActivationType Acti_Type;
    if(acti_type == "relu"){
        Acti_Type = ActivationType::kRELU;
    }
    else if(acti_type == "sigmoid")
    {
        Acti_Type = ActivationType::kSIGMOID;
    }
    else if (acti_type == "tanh") {
        Acti_Type = ActivationType::kTANH;
    }
    else if (acti_type == "elu") {
        Acti_Type = ActivationType::kELU;
    }
    else if (acti_type == "selu") {
        Acti_Type = ActivationType::kSELU;
    }
    else if (acti_type == "softsign") {
        Acti_Type = ActivationType::kSOFTSIGN;
    }
    else if (acti_type == "softplus") {
        Acti_Type = ActivationType::kSOFTPLUS;
    }
    else if (acti_type == "l_relu") {
        Acti_Type = ActivationType::kLEAKY_RELU;
    }
    else if (acti_type == "clip") {
        Acti_Type = ActivationType::kCLIP;
    }
    else if(acti_type == "hsigmoid"){
        Acti_Type = ActivationType::kHARD_SIGMOID;
    }
    else if(acti_type == "stanh"){
        Acti_Type = ActivationType::kSCALED_TANH;
    }
    else if(acti_type == "thres"){
        Acti_Type = ActivationType::kTHRESHOLDED_RELU;
    }
    else if (acti_type == "silu") {
        Acti_Type = ActivationType::kSIGMOID;
        auto acti = m_Network->addActivation(*input,Acti_Type);
        auto silu = m_Network->addElementWise(*input,*acti->getOutput(0),ElementWiseOperation::kPROD);
        return silu->getOutput(0);
    }
    else if(acti_type == "hardswish"){
        auto creator = getPluginRegistry()->getPluginCreator("HardSwishLayer_TRT", "1");
        const PluginFieldCollection* pluginData = creator->getFieldNames();
        IPluginV2 *pluginObj = creator->createPlugin("hardswish", pluginData);
        ITensor* inputTensors[] = {input};
        auto hs = m_Network->addPluginV2(inputTensors, 1, *pluginObj);
        return hs->getOutput(0);
    }
    else{
        cout<<"this active type is not support!"<<endl;
        assert(0);
    }
    IActivationLayer *acti = m_Network->addActivation(*input,Acti_Type);
    if(alpha != 0.f)
        acti ->setAlpha(alpha);
    if(beta != 0.f)
        acti ->setBeta(beta);

    return acti->getOutput(0);
}
ITensor* trt::trt_poolNet(ITensor* input, string pooltype, DimsHW kernel, DimsHW stride, DimsHW padding)
{
    PoolingType p_type;
    if(pooltype == "kMAX")
    {
        p_type = PoolingType::kMAX;
    }
    else if (pooltype == "kAVG") {
        p_type = PoolingType::kAVERAGE;
    }
    else {
        p_type = PoolingType::kMAX_AVERAGE_BLEND;
    }
    IPoolingLayer *poolLayer = m_Network->addPoolingNd(*input,p_type,kernel);
    assert(poolLayer);
    poolLayer->setStrideNd(stride);
    poolLayer->setPaddingNd(padding);
    return poolLayer->getOutput(0);
}
ITensor* trt::trt_eltNet(ITensor *input1, ITensor *input2,string elt_Type)
{
    ElementWiseOperation EltType;
    if (elt_Type == "kSUM")
    {
        EltType = ElementWiseOperation::kSUM;
    }
    else if(elt_Type == "kPROD") {
        EltType = ElementWiseOperation::kPROD;
    }
    else if (elt_Type == "kMAX") {
        EltType = ElementWiseOperation::kMAX;
    }
    else if (elt_Type == "kMIN") {
        EltType = ElementWiseOperation::kMIN;
    }
    else if (elt_Type == "kSUB") {
        EltType = ElementWiseOperation::kSUB;
    }
    else if (elt_Type == "kDIV"){
        EltType = ElementWiseOperation::kDIV;
    }
    else if (elt_Type == "kPOW") {
        EltType = ElementWiseOperation::kPOW;
    }
    else if (elt_Type == "kF_DIV") {
        EltType = ElementWiseOperation::kFLOOR_DIV;
    }
    else {
        cout<<"elt_Type is not found! please check it!"<<endl;
        assert(0);
    }
    IElementWiseLayer *elt = m_Network->addElementWise(*input1,*input2,EltType);
    return elt->getOutput(0);
}
ITensor* trt::trt_resnetCBA(Json::Value temp,ITensor* input)
{
    string convFile = param.weightPath + temp[0][0].asString() + ".weight.wgt";
    int output_c = temp[0][1].asInt();
    DimsHW kernel = DimsHW{1,1};
    DimsHW stride = DimsHW{1,1};
    DimsHW padding = DimsHW{0,0};
    DimsHW dilations = DimsHW{1,1};
    int groups = 1;
    if(temp[0].size()>2) kernel.h() = kernel.w() = temp[0][2].asInt();
    if(temp[0].size()>3) stride.h() = stride.w() = temp[0][3].asInt();
    if(temp[0].size()>4) padding.h() = padding.w() = temp[0][4].asInt();
    if(temp[0].size()>5) dilations.h() = dilations.w() = temp[0][5].asInt();
    if(temp[0].size()>6) groups = temp[0][6].asInt();
    cout<< "        outputC = "<<output_c<<",kernel="<<kernel.h()<<",stride="<<stride.h()<<",padding="<<padding.h()<<",dilations="<<dilations.h()
        <<",groups="<<groups<<endl;
    ITensor* output = trt_convNet(input,convFile,"",output_c,kernel,stride,padding,dilations,groups);
    string bnFile = param.weightPath + temp[1][0].asString();
    float eps = 1e-5;
    if(temp[1].size()>=2) eps = temp[1][1].asFloat();
    output = trt_bnNet(output,bnFile,eps);
    if(temp.size() == 2)
        return output;
    else if(temp.size() == 3){
        string activeType = temp[2][0].asString();
        float alpha = 0.0;
        float beta = 0.0;
        if(temp[2].size()>=2) alpha = temp[2][1].asFloat();
        if(temp[2].size()>=3) beta = temp[2][2].asFloat();
        output = trt_activeNet(output,activeType,alpha,beta);
        return output;
    }
    else{
        cout<<"Error! temp size just support 2 or 3,please check it!"<<endl;
        assert(0);
    }
}
void trt::trt_preInput(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    bool div_255 = layer["div_255"].asBool();
    Json::Value Mean = layer["Mean"];
    Json::Value Std = layer["Std"];
    int c = Layers[inputName]->getDimensions().d[0];
    Dims3 dimensions{c,1,1};
    ITensor* temp = Layers[inputName];
    if(div_255)
    {
        Weights Div_255{DataType::kFLOAT,nullptr,c};
        float *div = new float[c];
        for(int i = 0; i < c; i++){
            div[i] = 255.0;
        }
        Div_255.values = div;
        auto Div = m_Network->addConstant(dimensions,Div_255);
        temp = trt_eltNet(temp,Div->getOutput(0),"kDIV");
    }
    if(Mean.size() == c)
    {
        Weights mean{DataType::kFLOAT,nullptr,c};
        float *M = new float[c];
        for(int i = 0; i < c; i++){
            M[i] = Mean[i].asFloat();
        }
        mean.values = M;
        auto m = m_Network->addConstant(dimensions,mean);
        temp = trt_eltNet(temp,m->getOutput(0),"kSUB");
    }
    if(Std.size() == c)
    {
        Weights std{DataType::kFLOAT,nullptr,c};
        float *S = new float[c];
        for(int i = 0; i < c; i++){
            S[i] = Std[i].asFloat();
        }
        std.values = S;
        auto s = m_Network->addConstant(dimensions,std);
        temp = trt_eltNet(temp,s->getOutput(0),"kDIV");
    }
    Layers[layerName] = temp;
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }

}
void trt::trt_conv(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string weightFile = layer["weightFile"].asString();
    weightFile = param.weightPath + weightFile + ".weight.wgt";
    string biasFile = layer["biasFile"].asString();
    biasFile = param.weightPath + biasFile + ".bias.wgt";
    Json::Value m_param = layer["parameter"];
    //int input_c = m_param["input_c"].asInt();
    int output_c = m_param["output_c"].asInt();
    DimsHW kernel = DimsHW{1,1};
    DimsHW stride = DimsHW{1,1};
    DimsHW padding = DimsHW{0,0};
    DimsHW dilations = DimsHW{1,1};
    if(m_param["kernel"].size() == 2){
        kernel.h() = m_param["kernel"][0].asInt();
        kernel.w() = m_param["kernel"][1].asInt();
    }
    else if(m_param["kernel"].size() == 1){
        kernel.h() = kernel.w() = m_param["kernel"][0].asInt();
    }
    if(m_param["padding"].size() == 2){
        padding.h() = m_param["padding"][0].asInt();
        padding.w() = m_param["padding"][1].asInt();
    }
    else if(m_param["padding"].size() == 1){
        padding.h() = padding.w() = m_param["padding"][0].asInt();
    }
    if(m_param["stride"].size() == 2){
        stride.h() = m_param["stride"][0].asInt();
        stride.w() = m_param["stride"][1].asInt();
    }
    else if(m_param["stride"].size() == 1){
        stride.h() = stride.w() = m_param["stride"][0].asInt();
    }
    if(m_param["dilations"].size() == 2){
        dilations.h() = m_param["dilations"][0].asInt();
        dilations.w() = m_param["dilations"][1].asInt();
    }
    else if(m_param["dilations"].size() == 1){
        dilations.h() = dilations.w() = m_param["dilations"][0].asInt();
    }

    int groups = 1;
    if(m_param["groups"].asInt() > 1)
    {
        groups = m_param["groups"].asInt();
    }
    cout<<layerName<<" parameters : "<<output_c<<","<<kernel.h()<<","<<kernel.w()<<",";
    cout<<stride.h()<<","<<stride.w()<<","<<padding.h()<<","<<padding.w()<<" dilations: "<<dilations.h()<<","
       <<dilations.w()<<",groups = "<<groups<<endl;
    bool pre = m_param["pre"].asBool();
    bool post = m_param["post"].asBool();

    ITensor* conv = trt_convNet(Layers[inputName],weightFile,biasFile,output_c,kernel,stride,padding,dilations,groups,pre,post);
    Layers[layerName] = conv;
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_deconv(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string weightFile = layer["weightFile"].asString();
    weightFile = param.weightPath + weightFile + ".weight.wgt";
    string biasFile = layer["biasFile"].asString();
    biasFile = param.weightPath + biasFile + ".bias.wgt";
    Json::Value m_param = layer["parameter"];
    int output_c = m_param["output_c"].asInt();
    DimsHW kernel = DimsHW{1,1};
    DimsHW stride = DimsHW{1,1};
    DimsHW padding = DimsHW{0,0};
    DimsHW dilations = DimsHW{1,1};
    if(m_param["kernel"].size() == 2){
        kernel.h() = m_param["kernel"][0].asInt();
        kernel.w() = m_param["kernel"][1].asInt();
    }
    else if(m_param["kernel"].size() == 1){
        kernel.h() = kernel.w() = m_param["kernel"][0].asInt();
    }
    if(m_param["padding"].size() == 2){
        padding.h() = m_param["padding"][0].asInt();
        padding.w() = m_param["padding"][1].asInt();
    }
    else if(m_param["padding"].size() == 1){
        padding.h() = padding.w() = m_param["padding"][0].asInt();
    }
    if(m_param["stride"].size() == 2){
        stride.h() = m_param["stride"][0].asInt();
        stride.w() = m_param["stride"][1].asInt();
    }
    else if(m_param["stride"].size() == 1){
        stride.h() = stride.w() = m_param["stride"][0].asInt();
    }
    if(m_param["dilations"].size() == 2){
        dilations.h() = m_param["dilations"][0].asInt();
        dilations.w() = m_param["dilations"][1].asInt();
    }
    else if(m_param["dilations"].size() == 1){
        dilations.h() = dilations.w() = m_param["dilations"][0].asInt();
    }

    int groups = 1;
    if(m_param["groups"].asInt() > 1)
    {
        groups = m_param["groups"].asInt();
    }
    cout<<layerName<<" parameters : "<<output_c<<","<<kernel.h()<<","<<kernel.w()<<",";
    cout<<stride.h()<<","<<stride.w()<<","<<padding.h()<<","<<padding.w()<<" dilations: "<<dilations.h()<<","
       <<dilations.w()<<",groups = "<<groups<<endl;
    bool pre = m_param["pre"].asBool();
    bool post = m_param["post"].asBool();

    ITensor* conv = trt_deconvNet(Layers[inputName],weightFile,biasFile,output_c,kernel,stride,padding,dilations,groups,pre,post);
    Layers[layerName] = conv;
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_padding(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    int preP = layer["preP"].asInt();
    int postP = layer["postP"].asInt();
    auto padding = m_Network->addPaddingNd(*Layers[inputName],DimsHW{preP,preP},DimsHW{postP,postP});
    Layers[layerName] = padding->getOutput(0);
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_bn(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string weightFile = layer["weightFile"].asString();
    float eps = layer["eps"].asFloat();
    if(eps == 0.f)
        eps = 1.0e-5;
    weightFile = param.weightPath + weightFile;
    ITensor *batchNorm = trt_bnNet(Layers[inputName],weightFile,eps);
    Layers[layerName] = batchNorm;
    debug_print(Layers[layerName],layerName + " dim:");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_active(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string active_type = layer["active_type"].asString();
    float alpha = layer["alpha"].asFloat();
    float beta = layer["beta"].asFloat();
    ITensor* active = trt_activeNet(Layers[inputName],active_type,alpha,beta);
    Layers[layerName] = active;
    string outputName = layer["outputName"].asString();
    debug_print(Layers[layerName],layerName+" dim : ");
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_pool(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    Json::Value m_param = layer["parameter"];
    string pooltype = m_param["poolType"].asString();
    int kernel_H = m_param["kernel"][0].asInt();
    int kernel_W = m_param["kernel"][1].asInt();
    int padding_H = m_param["padding"][0].asInt();
    int padding_W = m_param["padding"][1].asInt();
    if (m_param.size() == 0)
    {
        padding_H = kernel_H / 2;
        padding_W = kernel_W / 2;
    }
    int stride_H = m_param["stride"][0].asInt();
    int stride_W = m_param["stride"][1].asInt();
    cout<<layerName<<"  parameter:"<<pooltype<<","<<kernel_H<<","<<kernel_W<<","
       <<stride_H<<","<<stride_W<<","<<padding_H<<","<<padding_W<<endl;

    DimsHW kernel = DimsHW{kernel_H,kernel_W};
    DimsHW stride = DimsHW{stride_H,stride_W};
    DimsHW padding = DimsHW{padding_H,padding_W};
    ITensor* poolLayer = trt_poolNet(Layers[inputName],pooltype,kernel,stride,padding);
    Layers[layerName] = poolLayer;
    debug_print(Layers[layerName],layerName+" dims :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_Pool(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    Json::Value m_param = layer["parameter"];
    string pooltype = m_param["poolType"].asString();
    int kernel_H = m_param["kernel"].asInt();
    int padding_H = kernel_H / 2;
    if(m_param["padding"].asInt() != 0)
        padding_H = m_param["padding"].asInt();

    int stride_H = m_param["stride"].asInt();
    cout<<layerName<<"  parameter:"<<pooltype<<","<<kernel_H<<","<<stride_H<<","<<padding_H<<endl;

    DimsHW kernel = DimsHW{kernel_H,kernel_H};
    DimsHW stride = DimsHW{stride_H,stride_H};
    DimsHW padding = DimsHW{padding_H,padding_H};
    ITensor* poolLayer = trt_poolNet(Layers[inputName],pooltype,kernel,stride,padding);
    Layers[layerName] = poolLayer;
    debug_print(Layers[layerName],layerName+" dims :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_elt(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    Json::Value Inputs = layer["inputName"];
    string elt_Type = layer["eltType"].asString();
    ITensor* elt = trt_eltNet(Layers[Inputs[0].asString()],Layers[Inputs[1].asString()],elt_Type);
    Layers[layerName] = elt;
    debug_print(Layers[layerName],layerName +" dims :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_fc(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string weightFile = layer["weightFile"].asString();
    weightFile = param.weightPath + weightFile + ".wgt";
    vector<float> fc_weights;
    vector<float> fc_bias;

    fc_weights = loadWeights(weightFile);
    string biasFile = layer["biasFile"].asString();
    biasFile = param.weightPath + biasFile + ".wgt";
    fc_bias = loadWeights(biasFile);
    Json::Value m_param = layer["parameter"];
    int input_c = m_param["input_c"].asInt();
    int output_c = m_param["output_c"].asInt();
    cout<<layerName<<" paramemter:"<<input_c<<","<<output_c<<endl;
    Weights groot{DataType::kFLOAT,nullptr,input_c*output_c};
    Weights racoon{DataType::kFLOAT,nullptr,output_c};
    float *grootWt = new float[input_c*output_c];
    for (int i =0;i<input_c*output_c;++i)
    {
        grootWt[i] = fc_weights.at(i);
    }
    groot.values = grootWt;
    float *racoonWt = new float[output_c];
    if(fc_bias.size() == output_c)
    {
        for (int j = 0;j<output_c;++j)
        {
            racoonWt[j] = fc_bias.at(j);
        }
    }
    else{
        for (int j = 0;j<output_c;++j) {
            racoonWt[j] = 0.0;
        }
    }
    racoon.values = racoonWt;
    IFullyConnectedLayer *fc = m_Network->addFullyConnected(*Layers[inputName],output_c,groot,racoon);
    assert(fc!=nullptr);
    fc_weights.clear();
    fc_bias.clear();
    Layers[layerName] = fc->getOutput(0);
    debug_print(Layers[layerName],layerName+" dim :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_concat(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    Json::Value Inputs = layer["inputName"];
    int n = Inputs.size();
    ITensor **b = new ITensor*[n];
    for (int i =0;i<n;i++) {
        b[i] = Layers[Inputs[i].asString()];
    }
    IConcatenationLayer *concat = m_Network->addConcatenation(b,n);
    int axis = layer["axis"].asInt();
    concat->setAxis(axis);
    Layers[layerName] = concat->getOutput(0);
    debug_print(Layers[layerName],layerName +" dims :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_slice(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    Json::Value start = layer["start"];
    Json::Value size = layer["size"];
    Json::Value stride = layer["stride"];
    Dims Start ;
    Dims Size;
    Dims Stride;
    Start.nbDims = start.size();
    Size.nbDims = size.size();
    Stride.nbDims = stride.size();
    for(unsigned int i = 0;i < start.size();i++)
    {
        Start.d[i] = start[i].asInt();
        Size.d[i] = size[i].asInt();
        Stride.d[i] = stride[i].asInt();
    }
    ISliceLayer *slice = m_Network->addSlice(*Layers[inputName],Start,Size,Stride);
    Layers[layerName] = slice->getOutput(0);
    debug_print(Layers[layerName],"Slice dim :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_softmax(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    ISoftMaxLayer *softmax = m_Network->addSoftMax(*Layers[inputName]);
    uint32_t axes = layer["axes"].asUInt();
    softmax->setAxes(axes);
    Layers[layerName] = softmax->getOutput(0);
    debug_print(Layers[layerName],layerName +" dims :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_shuffle(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    bool isReshape = layer["isReshape"].asBool();
    bool reshapeFirst = layer["reshapeFirst"].asBool();
    bool isPermute = layer["isPermute"].asBool();
    Dims dimensions;
    Permutation permutation;
    if(isReshape)
    {
        Json::Value reshape = layer["reshape"];
        int size = reshape.size();
        dimensions.nbDims = size;
        for (int i =0;i<size;i++) {
           dimensions.d[i] = reshape[i].asInt();
        }
    }
    if(isPermute)
    {
        Json::Value permute = layer["permute"];
        int permuteSize = permute.size();
        for (int i = 0;i<permuteSize;i++)
        {
            permutation.order[i] = permute[i].asInt();
        }
    }
    IShuffleLayer *shuffle = m_Network->addShuffle(*Layers[inputName]);
    if(isReshape && isPermute)
    {
        if(reshapeFirst)
        {
            shuffle->setReshapeDimensions(dimensions);
            shuffle->setSecondTranspose(permutation);
        }
        else {
            shuffle->setFirstTranspose(permutation);
            shuffle->setReshapeDimensions(dimensions);
        }
    }
    else if(!isReshape && isPermute)
    {
        shuffle->setFirstTranspose(permutation);
    }
    else if(isReshape && !isPermute)
    {
        shuffle->setReshapeDimensions(dimensions);
    }
    Layers[layerName] = shuffle->getOutput(0);
    debug_print(Layers[layerName],"Shuffle dim : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_matmul(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    Json::Value inputName = layer["inputName"];
    Json::Value matrixType = layer["matrixType"];
    MatrixOperation matrix1Type,matrix2Type;
    if (matrixType.size() != 2)
    {
        matrix1Type = MatrixOperation::kNONE;
        matrix2Type = MatrixOperation::kNONE;
    }
    else {
        if (matrixType[0].asString() == "kNONE" || matrixType[0].asString().size() == 0)
        {
            matrix1Type = MatrixOperation::kNONE;
        }
        else
        {
            matrix1Type = MatrixOperation::kTRANSPOSE;
        }
        if (matrixType[1].asString() == "kNONE" || matrixType[1].asString().size() == 0)
        {
            matrix2Type = MatrixOperation::kNONE;
        }
        else {
            matrix2Type = MatrixOperation::kTRANSPOSE;
        }
    }
    debug_print(Layers[inputName[0].asString()],"intput 0 dims :");
    debug_print(Layers[inputName[1].asString()],"intput 1 dims :");
    IMatrixMultiplyLayer *matmul = m_Network->addMatrixMultiply(*Layers[inputName[0].asString()],matrix1Type,
                                                                     *Layers[inputName[1].asString()],matrix2Type);

    Layers[layerName] = matmul->getOutput(0);
    debug_print(Layers[layerName],"Matmul dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_topk(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string TopKOperation = layer["TopKOperation"].asString();
    nvinfer1::TopKOperation op;
    if (TopKOperation == "kMAX")
    {
        op = TopKOperation::kMAX;
    }
    else if (TopKOperation == "kMIN") {
        op = TopKOperation::kMIN;
    }
    else {
        cout<< "TopKOperation type is error! please check it!"<<endl;
        return;
    }
    int k = layer["k"].asInt();
    uint32_t reduceAxes = layer["reduceAxes"].asUInt();
    ITopKLayer *topK = m_Network->addTopK(*Layers[inputName],op,k,reduceAxes);
    int outputIndex = layer["outputIndex"].asInt();
    Layers[layerName] = topK->getOutput(outputIndex);
    debug_print(Layers[layerName],"topK dim :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_reduce(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string type = layer["dtype"].asString();
    ReduceOperation op;
    if(type == "kMAX")
        op = ReduceOperation::kMAX;
    else if(type == "kMIN")
        op = ReduceOperation::kMIN;
    else if(type == "kSUM")
        op = ReduceOperation::kSUM;
    else if(type == "kPROD")
        op = ReduceOperation::kPROD;
    else
        op = ReduceOperation::kAVG;
    uint32_t axes = layer["axes"].asUInt();
    bool keepD = layer["keepD"].asBool();
    IReduceLayer *reduce = m_Network->addReduce(*Layers[inputName],op,axes,keepD);
    Layers[layerName] = reduce->getOutput(0);
    debug_print(Layers[layerName],"reduce dim :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_constant(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    Json::Value dims = layer["dims"];
    Json::Value dtype = layer["dtype"];
    Dims dimension;
    dimension.nbDims = dims.size();
    int allNum = 1;
    for (int i = 0; i < dimension.nbDims; i++)
    {
        dimension.d[i] = dims[i].asInt();
        allNum *= dims[i].asInt();
    }
    IConstantLayer *out;
    if (dtype.asString() == "KINT32")
    {
        int alpha = layer["alpha"].asInt();
        Weights wgt{DataType::kINT32, nullptr, allNum};
        int *w = new int[allNum];
        for (int i = 0; i < allNum; i++)
        {
            w[i] = alpha;
        }
        wgt.values = w;
        out = m_Network->addConstant(dimension, wgt);
    }
    else
    {
        float alpha = layer["alpha"].asFloat();
        Weights wgt{DataType::kFLOAT, nullptr, allNum};
        float *w = new float[allNum];
        for (int i = 0; i < allNum; i++)
        {
            w[i] = alpha;
        }
        wgt.values = w;
        out = m_Network->addConstant(dimension, wgt);
    }

    Layers[layerName] = out->getOutput(0);
    debug_print(Layers[layerName], "dims :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_pReLU(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string preluFile = layer["preluFile"].asString();
    vector<float> param_weight;
    preluFile = param.weightPath + preluFile + ".wgt";
    param_weight = loadWeights(preluFile);
    int channel = Layers[inputName]->getDimensions().d[0];
    Weights P_wgt{DataType::kFLOAT,nullptr,channel};
    float *var = new float[channel];
    for(int i = 0; i< channel; i++)
    {
        var[i] = param_weight[i];
    }
    P_wgt.values = var;
    auto slope = m_Network->addConstant(Dims3(channel,1,1),P_wgt);
    IParametricReLULayer *PReLU = m_Network->addParametricReLU(*Layers[inputName],*slope->getOutput(0));
    Layers[layerName] =  PReLU->getOutput(0);
    debug_print(Layers[layerName],"reduce dim :");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }

}
void trt::trt_convBnActive(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string weightFile = layer["convFile"].asString();
    string weightPath = param.weightPath + weightFile;
    weightFile = weightPath + ".weight.wgt";
    string biasFile = layer["biasFile"].asString();
    biasFile = weightPath + ".bias.wgt";
    Json::Value m_param = layer["parameter"];
    int output_c = m_param["output_c"].asInt();
    DimsHW kernel = DimsHW{1,1};
    DimsHW stride = DimsHW{1,1};
    DimsHW padding = DimsHW{0,0};
    DimsHW dilations = DimsHW{1,1};
    if(m_param["kernel"].size() == 2){
        kernel.h() = m_param["kernel"][0].asInt();
        kernel.w() = m_param["kernel"][1].asInt();
    }
    else if(m_param["kernel"].size() == 1){
        kernel.h() = kernel.w() = m_param["kernel"][0].asInt();
    }
    if(m_param["padding"].size() == 2){
        padding.h() = m_param["padding"][0].asInt();
        padding.w() = m_param["padding"][1].asInt();
    }
    else if(m_param["padding"].size() == 1){
        padding.h() = padding.w() = m_param["padding"][0].asInt();
    }
    if(m_param["stride"].size() == 2){
        stride.h() = m_param["stride"][0].asInt();
        stride.w() = m_param["stride"][1].asInt();
    }
    else if(m_param["stride"].size() == 1){
        stride.h() = stride.w() = m_param["stride"][0].asInt();
    }
    if(m_param["dilations"].size() == 2){
        dilations.h() = m_param["dilations"][0].asInt();
        dilations.w() = m_param["dilations"][1].asInt();
    }
    else if(m_param["dilations"].size() == 1){
        dilations.h() = dilations.w() = m_param["dilations"][0].asInt();
    }
    int groups = 1;
    if(m_param["groups"].asInt() > 1)
    {
        groups = m_param["groups"].asInt();
    }
    cout<<layerName<<" parameters : "<<output_c<<","<<kernel.h()<<","<<kernel.w()<<",";
    cout<<stride.h()<<","<<stride.w()<<","<<padding.h()<<","<<padding.w()
       <<" dilations: "<<dilations.h()<<","<<dilations.w()<<",groups ="<<groups<<endl;

    ITensor* output = trt_convNet(Layers[inputName],weightFile,biasFile,output_c,kernel,stride,padding,dilations,groups);
    string bnFile = layer["bnFile"].asString();
    if (bnFile.size() > 0){
        bnFile = param.weightPath + bnFile;
        float eps = 1.0e-5;
        if(layer["eps"].asFloat() > 0.f)
            eps = layer["eps"].asFloat();
        output = trt_bnNet(output,bnFile,eps);
    }
    string active_type = layer["active_type"].asString();
    if (active_type.size() > 0){
        float alpha = layer["alpha"].asFloat();
        float beta = layer["beta"].asFloat();
        output = trt_activeNet(output,active_type,alpha,beta);
    }
    Layers[layerName] = output;
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_resnetLayer(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    Json::Value left = layer["left"];
    Json::Value right = layer["right"];
    ITensor* Left = Layers[inputName];
    cout<< layerName<<" : resnet param:"<<endl;
    for(int i = 0;i<left.size();i++)
    {
        Json::Value temp = left[i];
        Left = trt_resnetCBA(temp,Left);
    }
    ITensor* Right = Layers[inputName];
    for(int i = 0;i<right.size();i++)
    {
        Json::Value temp = right[i];
        Right = trt_resnetCBA(temp,Right);
    }
    ITensor* output = m_Network->addElementWise(*Left,*Right,ElementWiseOperation::kSUM)->getOutput(0);
    assert(output);
    string activeType = layer["active_type"].asString();
    if(activeType.size() > 0){
        float alpha = layer["alpha"].asFloat();
        float beta = layer["beta"].asFloat();
        output = trt_activeNet(output,activeType,alpha,beta);
        assert(output);
    }
    Layers[layerName] = output;
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }

}
void trt::trt_resnet3(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    bool downsample = layer["downsample"].asBool();
    string weightFile = layer["weightsFile"].asString();
    string weightPath = param.weightPath + weightFile;
    weightFile = weightPath + ".conv1.weight.wgt";
    string biasFile ;
    biasFile = weightPath + ".conv1.bias.wgt";
    Json::Value m_param = layer["parameter"];
    //int input_c = m_param["input_c"].asInt();
    int temp_c = m_param["temp_c"].asInt();
    int output_c = m_param["output_c"].asInt();
    ITensor* output1 = trt_convNet(Layers[inputName],weightFile,biasFile,temp_c,DimsHW{1,1},DimsHW{1,1},DimsHW{0,0},DimsHW{1,1});
    string bnFile ;
    bnFile = weightPath + ".bn1";
    output1 = trt_bnNet(output1,bnFile);
    IActivationLayer *acti = m_Network->addActivation(*output1,ActivationType::kRELU);
    output1 = acti->getOutput(0);
    weightFile = weightPath + ".conv2.weight.wgt";
    biasFile = weightPath + ".conv2.bias.wgt";
    int temp_s = 1;
    if(downsample)
    {
        temp_s = 2;
    }
    output1 = trt_convNet(output1,weightFile,biasFile,temp_c,DimsHW{3,3},DimsHW{temp_s,temp_s},DimsHW{1,1},DimsHW{1,1});
    bnFile = weightPath + ".bn2";
    output1 = trt_bnNet(output1,bnFile);
    acti = m_Network->addActivation(*output1,ActivationType::kRELU);
    output1 = acti->getOutput(0);
    weightFile = weightPath + ".conv3.weight.wgt";
    biasFile = weightPath + ".conv3.bias.wgt";
    output1 = trt_convNet(output1,weightFile,biasFile,output_c,DimsHW{1,1},DimsHW{1,1},DimsHW{0,0},DimsHW{1,1});
    bnFile = weightPath + ".bn3";
    output1 = trt_bnNet(output1,bnFile);
    ITensor* output2 = Layers[inputName];
    if(downsample)
    {
        weightFile = weightPath + ".downsample.0.weight.wgt";
        biasFile = weightPath + ".downsample.0.bias.wgt";
        output2 = trt_convNet(Layers[inputName],weightFile,biasFile,output_c,DimsHW{1,1},DimsHW{temp_s,temp_s},DimsHW{0,0},DimsHW{1,1});
        bnFile = weightPath + ".downsample.1";
        output2 = trt_bnNet(output2,bnFile);
    }
    auto cat = m_Network->addElementWise(*output1,*output2,ElementWiseOperation::kSUM);
    auto result = m_Network->addActivation(*cat->getOutput(0),ActivationType::kRELU);
    Layers[layerName] = result->getOutput(0);
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_focus(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    auto s1 = m_Network->addSlice(*Layers[inputName], Dims3{0, 0, 0}, Dims3{param.input_c, param.input_h / 2, param.input_w / 2}, Dims3{1, 2, 2});
    assert(s1);
    auto s2 = m_Network->addSlice(*Layers[inputName], Dims3{0, 1, 0}, Dims3{param.input_c, param.input_h / 2, param.input_w / 2}, Dims3{1, 2, 2});
    assert(s2);
    auto s3 = m_Network->addSlice(*Layers[inputName], Dims3{0, 0, 1}, Dims3{param.input_c, param.input_h / 2, param.input_w / 2}, Dims3{1, 2, 2});
    assert(s3);
    auto s4 = m_Network->addSlice(*Layers[inputName], Dims3{0, 1, 1}, Dims3{param.input_c, param.input_h / 2, param.input_w / 2}, Dims3{1, 2, 2});
    assert(s4);

    ITensor* inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)};
    auto cat = m_Network->addConcatenation(inputTensors, 4);
    cat->setAxis(0);
    Layers[layerName] = cat->getOutput(0);
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }

}
void trt::trt_UpSample(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string mode = layer["mode"].asString();
    ResizeMode smode;
    if(mode == "nearest")
    {
        smode = ResizeMode::kNEAREST;
    }
    else{
        smode = ResizeMode::kLINEAR;
    }
    auto out = m_Network->addResize(*Layers[inputName]);
    Dims3 dims;
    dims.d[0] = Layers[inputName]->getDimensions().d[0];
    Json::Value grid = layer["grid"];

    if(grid.size() == 0)
    {
        dims.d[1] = Layers[inputName]->getDimensions().d[1]*2;
        dims.d[2] = Layers[inputName]->getDimensions().d[2]*2;
    }
    else if(grid.size() == 2){
        dims.d[1] = grid[0].asInt();
        dims.d[2] = grid[1].asInt();
    }
    else{
        dims.d[1] = grid[0].asInt();
        dims.d[2] = grid[0].asInt();
    }
    out->setOutputDimensions(dims);
    out->setResizeMode(smode);
    if(smode == ResizeMode::kLINEAR){
        out->setAlignCorners(true);
    }
    Layers[layerName] = out->getOutput(0);
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_UpSample_plugin(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    Json::Value grid = layer["grid"];

    float scale_h = 2.0f;
    float scale_w = 2.0f;
    if(grid.size() == 2)
    {
        scale_h = 1.0*grid[0].asInt() / Layers[inputName]->getDimensions().d[1];
        scale_w = 1.0*grid[1].asInt() / Layers[inputName]->getDimensions().d[2];
    }
    else if(grid.size() == 1){
        scale_h = 1.0*grid[0].asInt() / Layers[inputName]->getDimensions().d[1];
        scale_w = 1.0*grid[0].asInt() / Layers[inputName]->getDimensions().d[2];
    }
    auto creator = getPluginRegistry()->getPluginCreator("UpsamplePlugin", "1");
    PluginField pField[1];
    float *s = new float[2];
    s[0] = scale_h;
    s[1] = scale_w;
    pField[0].data = s;
    pField[0].length = 1;
    pField[0].type = PluginFieldType::kFLOAT32;
    pField[0].name = "scaleFactor";

    PluginFieldCollection pluginData;
    pluginData.nbFields = 1;
    pluginData.fields = pField;
    //string lname = "upSample";
    IPluginV2 *pluginObj = creator->createPlugin(layerName.c_str(), &pluginData);
    ITensor* inputTensors[] = {Layers[inputName]};
    auto upS = m_Network->addPluginV2(inputTensors, 1, *pluginObj);
    Layers[layerName] = upS->getOutput(0);
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }

}
void trt::trt_groupNorm(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    int groups = layer["groups"].asInt();
    int a = Layers[inputName]->getDimensions().d[0]%groups;
    if(a!=0)
    {
        cout<< "input channel % groups  != 0 "<<endl;
        assert(0);
    }
    Dims dimensions;
    dimensions.nbDims = 2;
    dimensions.d[0] = groups;
    dimensions.d[1] = -1;
    IShuffleLayer *shuffle1 = m_Network->addShuffle(*Layers[inputName]);
    shuffle1->setReshapeDimensions(dimensions);  //[c,h,w] -->[groups,c*h*w/groups]
    IReduceLayer *mean = m_Network->addReduce(*shuffle1->getOutput(0),ReduceOperation::kAVG,2,true);//mean : [groups,1]
    IElementWiseLayer *elt_sub = m_Network->addElementWise(*shuffle1->getOutput(0),*mean->getOutput(0),
                                                                     ElementWiseOperation::kSUB);   // x - mean
    IElementWiseLayer *elt_prod = m_Network->addElementWise(*elt_sub->getOutput(0),*elt_sub->getOutput(0),
                                                                      ElementWiseOperation::kPROD); // var**2 = (x-mean)**2
    IReduceLayer *var_var =  m_Network->addReduce(*elt_prod->getOutput(0),ReduceOperation::kSUM,2,true);//sum(var**2) :[groups,1]
    Weights E{DataType::kFLOAT,nullptr,groups};
    float* e = new float[groups];
    for (int i =0; i< groups; i++)
    {
        e[i] = 1e-5;
    }
    E.values = e;
    IConstantLayer *e_layer = m_Network->addConstant(Dims2{groups,1},E);//1e-5
    float* length = new float[groups];
    for (int i =0; i< groups; i++)
    {
        length[i] = shuffle1->getOutput(0)->getDimensions().d[1]*1.0;
    }
    E.values = length;
    IConstantLayer *e_length = m_Network->addConstant(Dims2{groups,1},E);//length = c*h*w/32
    IElementWiseLayer *elt_div = m_Network->addElementWise(*var_var->getOutput(0),*e_length->getOutput(0),
                                                                     ElementWiseOperation::kDIV); //sum(var**2) / length
    IElementWiseLayer *elt_sum = m_Network->addElementWise(*elt_div->getOutput(0),*e_layer->getOutput(0),
                                                                     ElementWiseOperation::kSUM); // 1e-5 + sum(var**2)/length
    int n = elt_sub->getOutput(0)->getDimensions().d[1];
    float* sqrt = new float[groups*n];
    Weights S{DataType::kFLOAT,nullptr,groups*n};
    for (int i =0; i< groups*n; i++)
    {
        sqrt[i] = 0.5;
    }
    S.values = sqrt;
    IConstantLayer *e_sqrt = m_Network->addConstant(Dims2{groups,n},S);//
    IElementWiseLayer *elt_sqrt = m_Network->addElementWise(*elt_sum->getOutput(0),*e_sqrt->getOutput(0),
                                                                      ElementWiseOperation::kPOW);//var = sqrt(1e-5 + sum(var**2)/length)

//    auto creator = getPluginRegistry()->getPluginCreator("divLayer_TRT", "1");
//    const PluginFieldCollection* pluginData = creator->getFieldNames();
//    IPluginV2 *pluginObj = creator->createPlugin(inputName.c_str(), pluginData);
//    ITensor* inputTensors[] = {elt_sub->getOutput(0),elt_sqrt->getOutput(0)};
//    auto div_var = m_Network->addPluginV2(inputTensors, 2, *pluginObj);

    IElementWiseLayer *div_var = m_Network->addElementWise(*elt_sub->getOutput(0),*elt_sqrt->getOutput(0),
                                                                     ElementWiseOperation::kDIV);//(x-mean)/var
    IShuffleLayer *shuffle2 = m_Network->addShuffle(*div_var->getOutput(0));//[groups,c*h*w/groups] -->[c,h,w]
    Dims dimensions2;
    dimensions2 = Layers[inputName]->getDimensions();
    shuffle2->setReshapeDimensions(dimensions2);//[groups,c*h*w/groups] -->[c,h,w]
    Weights scale{DataType::kFLOAT,nullptr,Layers[inputName]->getDimensions().d[0]};
    Weights bias{DataType::kFLOAT,nullptr,Layers[inputName]->getDimensions().d[0]};
    float *Scale_weight = new float[Layers[inputName]->getDimensions().d[0]];
    float *Bias_weight = new float[Layers[inputName]->getDimensions().d[0]];
    vector<float> scale_weight;
    vector<float> sias_weight;
    int c_dim = Layers[inputName]->getDimensions().d[0];
    string Scale_file = layer["scale"].asString();
    string Bias_flie = layer["bias"].asString();
    Scale_file = param.weightPath + Scale_file + ".wgt";
    Bias_flie = param.weightPath + Bias_flie + ".wgt";
    scale_weight = loadWeights(Scale_file);
    sias_weight = loadWeights(Bias_flie);
    for(int i = 0;i<c_dim;i++)
    {
        Scale_weight[i] = scale_weight[i];
        Bias_weight[i] = sias_weight[i];
    }
    scale.values = Scale_weight;
    bias.values = Bias_weight;

    IConstantLayer *add_scale = m_Network->addConstant(Dims3{c_dim,1,1},scale);//scale
    IElementWiseLayer *prod_scale = m_Network->addElementWise(*shuffle2->getOutput(0),*add_scale->getOutput(0),
                                                                        ElementWiseOperation::kPROD);//x * scale
    IConstantLayer *add_bias = m_Network->addConstant(Dims3{c_dim,1,1},bias);//bias
    IElementWiseLayer *sum_bias = m_Network->addElementWise(*prod_scale->getOutput(0),*add_bias->getOutput(0),
                                                                      ElementWiseOperation::kSUM);//x * scale + bias
    Layers[layerName] = sum_bias->getOutput(0);
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::trt_unary(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    string unary_type = layer["unary_type"].asString();
    UnaryOperation op;
    if(unary_type == "kEXP")
    {
        op = UnaryOperation::kEXP;
    }
    else if(unary_type == "kLOG"){
        op = UnaryOperation::kLOG;
    }
    else if(unary_type == "kSQRT"){
        op = UnaryOperation::kSQRT;
    }
    else if(unary_type == "kRECIP"){
        op = UnaryOperation::kRECIP;
    }
    else if(unary_type == "kABS"){
        op = UnaryOperation::kABS;
    }
    else if(unary_type == "kNEG"){
        op = UnaryOperation::kNEG;
    }
    else if(unary_type == "kSIN"){
        op = UnaryOperation::kSIN;
    }
    else if(unary_type == "kCOS"){
        op = UnaryOperation::kCOS;
    }
    else if(unary_type == "kTAN"){
        op = UnaryOperation::kTAN;
    }
    else if(unary_type == "kSINH"){
        op = UnaryOperation::kSINH;
    }
    else if(unary_type == "kCOSH"){
        op = UnaryOperation::kCOSH;
    }
    else if(unary_type == "kASIN"){
        op = UnaryOperation::kASIN;
    }
    else if(unary_type == "kACOS"){
        op = UnaryOperation::kACOS;
    }
    else if(unary_type == "kATAN"){
        op = UnaryOperation::kATAN;
    }
    else if(unary_type == "kASINH"){
        op = UnaryOperation::kASINH;
    }
    else if(unary_type == "kACOSH"){
        op = UnaryOperation::kACOSH;
    }
    else if(unary_type == "kATANH"){
        op = UnaryOperation::kATANH;
    }
    else if(unary_type == "kCEIL"){
        op = UnaryOperation::kCEIL;
    }
    else if(unary_type == "kFLOOR"){
        op = UnaryOperation::kFLOOR;
    }
    else{
        cout<<"no thie unary type! please check it!"<<endl;
        assert(0);
    }
    IUnaryLayer *unary = m_Network->addUnary(*Layers[inputName],op);
    Layers[layerName] = unary->getOutput(0);
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
ITensor* trt::convBlock(ITensor *input, int outch, int k, int s, string lname,string acti_type,
                        float eps,float alpha)
{
    int p = k / 2;
    ITensor* out = trt_convNet(input,lname + ".conv.weight.wgt","",outch,DimsHW{k,k},DimsHW{s,s},DimsHW{p,p});
    out = trt_bnNet(out,lname + ".bn", eps);
    out = trt_activeNet(out,acti_type,alpha);
    return out;
}
ITensor* trt::bottleneck(ITensor *input, string lname, string acti_type, int c1, int c2, bool shortcut, float e,
                         float eps, float alpha)
{
    ITensor* cv1 = convBlock(input,(int)((float)c2 * e),1,1,lname+".cv1",acti_type,eps,alpha);
    ITensor* cv2 = cv2 = convBlock(cv1, c2, 3, 1, lname + ".cv2",acti_type,eps,alpha);
    if (shortcut && c1 == c2) {
        auto ew = trt_eltNet(input,cv2,"kSUM");
        return ew;
    }
    return cv2;
}
void trt::yolo_C3(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    int c1 = layer["c1"].asInt();
    int c2 = layer["c2"].asInt();
    int n = layer["n"].asInt();
    bool SC = layer["shortCut"].asBool();
    int g = layer["g"].asInt();
    float e = layer["e"].asFloat();
    string lname = layer["lname"].asString();
    lname = param.weightPath + lname;
    string acti_type = layer["active_type"].asString();
    float eps = layer["eps"].asFloat();
    if(eps == 0.f)
        eps = 1e-3;
    float alpha = layer["alpha"].asFloat();
    if(alpha == 0.f)
        alpha = 0.1;
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(Layers[inputName], c_, 1, 1,lname + ".cv1",acti_type,eps,alpha);
    auto cv2 = convBlock(Layers[inputName], c_, 1, 1,lname + ".cv2",acti_type,eps,alpha);
    ITensor *y1 = cv1;
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(y1, lname + ".m." + std::to_string(i),acti_type,c_, c_, SC, 1.0,eps,alpha);
        y1 = b;
    }
    ITensor* inputTensors[] = { y1, cv2 };
    auto cat = m_Network->addConcatenation(inputTensors, 2);
    auto cv3 = convBlock(cat->getOutput(0), c2, 1, 1,lname + ".cv3",acti_type,eps,alpha);
    Layers[layerName] = cv3;
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
void trt::yolo_spp(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    string inputName = layer["inputName"].asString();
    Json::Value kernels = layer["kernels"];
    int c1 = layer["c1"].asInt();
    int c_ = c1 / 2;
    int c2 = layer["c2"].asInt();
    string lname = layer["lname"].asString();
    lname  = param.weightPath + lname;
    string acti_type = layer["active_type"].asString();
    float eps = layer["eps"].asFloat();
    if(eps == 0.f)
        eps = 1e-3;
    float alpha = layer["alpha"].asFloat();
    if(alpha == 0.f)
        alpha = 0.1;
    ITensor** spp = new ITensor*[4];
    ITensor* cv1 = convBlock(Layers[inputName],c_,1,1,lname+".cv1",acti_type,eps,alpha);
    spp[0] = cv1;
    for(int i = 1;i< 4;i++)
    {
        int k = kernels[i-1].asInt();
        spp[i] = trt_poolNet(cv1,"kMAX",DimsHW{k,k},DimsHW{1,1},DimsHW{k/2,k/2});
    }
    ITensor* cat = m_Network->addConcatenation(spp,4)->getOutput(0);
    ITensor* cv2 = convBlock(cat,c2,1,1,lname+".cv2",acti_type,eps,alpha);
    //debug_print(cv2,"spp");
    Layers[layerName] = cv2;
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }

}
void trt::trt_yolo(Json::Value layer)
{
    string layerName = layer["layerName"].asString();
    Json::Value inputs = layer["inputName"];
    string anchor_grid = layer["anchor_grid"].asString();
    vector<float> anchors_yolo = loadWeights(param.weightPath + anchor_grid + ".wgt");
//    for(int i = 0 ;i < anchors_yolo.size();i++)
//    {
//        cout<<anchors_yolo[i]<< "  ,";
//    }
//    cout<<endl;
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    PluginField pluginMultidata[4];
    int NetData[4];
    NetData[0] = layer["cls_num"].asInt();
    NetData[1] = param.input_w;
    NetData[2] = param.input_h;
    NetData[3] = layer["max_box"].asInt();
    pluginMultidata[0].data = NetData;
    pluginMultidata[0].length = 4;
    pluginMultidata[0].name = "netdata";
    pluginMultidata[0].type = PluginFieldType::kFLOAT32;
    int scale[3] = { 8, 16, 32 };
    int plugindata[3][8];
    string names[3];
    for (int k = 1; k < 4; k++)
    {
        plugindata[k - 1][0] = param.input_w / scale[k - 1];
        plugindata[k - 1][1] = param.input_h / scale[k - 1];
        for (int i = 2; i < 8; i++)
        {
            plugindata[k - 1][i] = int(anchors_yolo[(k - 1) * 6 + i - 2]);
        }
        pluginMultidata[k].data = plugindata[k - 1];
        pluginMultidata[k].length = 8;
        names[k - 1] = "yolodata" + std::to_string(k);
        pluginMultidata[k].name = names[k - 1].c_str();
        pluginMultidata[k].type = PluginFieldType::kFLOAT32;
    }
    PluginFieldCollection pluginData;
    pluginData.nbFields = 4;
    pluginData.fields = pluginMultidata;
    IPluginV2 *pluginObj = creator->createPlugin(layerName.c_str(), &pluginData);
    ITensor* inputTensors_yolo[] = { Layers[inputs[2].asString()], Layers[inputs[1].asString()], Layers[inputs[0].asString()] };
    auto yolo = m_Network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    Layers[layerName] = yolo->getOutput(0);
    debug_print(Layers[layerName],layerName +" dims : ");
    string outputName = layer["outputName"].asString();
    if (outputName.size() > 0)
    {
        Layers[layerName]->setName(outputName.c_str());
        m_Network->markOutput(*Layers[layerName]);
    }
}
