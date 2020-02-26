#'alexnet' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'
#MODELS=('resnext50_32x4d' 'resnext101_32x8d' 'wide_resnet50_2' 'wide_resnet101_2' 'vgg11' 'vgg11_bn' 'vgg13' 'vgg13_bn' 'vgg16' 'vgg16_bn' 'vgg19_bn' 'vgg19' 'squeezenet1_0' 'squeezenet1_1' 'inception_v3' 'densenet121' 'densenet169' 'densenet201' 'densenet161' 'googlenet' 'mobilenet_v2' 'mnasnet0_5' 'mnasnet0_75' 'mnasnet1_0' 'mnasnet1_3' 'shufflenet_v2_x0_5' 'shufflenet_v2_x1_0' 'shufflenet_v2_x1_5' 'shufflenet_v2_x2_0')
#'googlenet' 'inception_v3'
#MODELS=('googlenet' 'inception_v3' 'alexnet' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152')
#MODELS=('fcn_resnet50' 'fcn_resnet101' 'deeplabv3_resnet50' 'deeplabv3_resnet101')
#MODELS=('deeplabv3_resnet50' 'deeplabv3_resnet101')
MODELS=('fasterrcnn_resnet50_fpn' 'maskrcnn_resnet50_fpn' 'keypointrcnn_resnet50_fpn')
for model in ${MODELS[@]}; do
    python test_run_model.py $model
done

