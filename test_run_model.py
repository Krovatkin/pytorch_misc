import torch
from torchvision import models
from test_jit import get_execution_plan
from common_utils import enable_profiling_mode
import sys
import argparse



def get_plan(model):
    ##fwd = model._c._get_method('forward')
    ##fwd.get_debug_state()
    state = model._c.get_debug_state()
    plan = get_execution_plan(state)
    num_bailouts = plan.code.num_bailouts()
    return plan

# https://github.com/pytorch/vision/blob/master/test/test_models.py#L18

def get_available_classification_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]

def do_segmentation_test(model_name):
    with enable_profiling_mode():
    # passing num_class equal to a number other than 1000 helps in making the test
    # more enforcing in nature
        print("testing ", model_name)
        torch.manual_seed(0)
        open("segmentation_{}".format(model_name), 'a').close()
        model = models.segmentation.__dict__[model_name](num_classes=50, pretrained_backbone=False)
        model.eval()
        scripted_model = torch.jit.script(model)
        input_shape = (1, 3, 300, 300)
        x = torch.rand(input_shape)
        # out = model(x)
        scripted_model(x)
        scripted_model(x)
        plan = get_plan(scripted_model)
        num_bailouts = plan.code.num_bailouts()
        print(num_bailouts)
        for i in range(0, num_bailouts):
            plan.code.request_bailout(i)
            bailout_output = scripted_model(x)


def do_detection_test(name):
    with enable_profiling_mode():
        torch.manual_seed(0)
        print("testing ", name)
        model = models.detection.__dict__[name](num_classes=50, pretrained_backbone=False)
        model.eval()
        input_shape = (3, 300, 300)
        x = torch.rand(input_shape)
        model_input = [x]
        out = model(model_input)
        scripted_model = torch.jit.script(model)
        scripted_model(model_input)
        scripted_model(model_input)
        plan = get_plan(scripted_model)
        num_bailouts = plan.code.num_bailouts()
        print(num_bailouts)
        for i in range(0, num_bailouts):
            plan.code.request_bailout(i)
            bailout_output = scripted_model(model_input)
        open("detection_{}".format(name), 'a').close()


def do_classification_test(model_name):
    with enable_profiling_mode():
        input_shape = (1, 3, 224, 224)
        if model_name in ['inception_v3']:
            input_shape = (1, 3, 299, 299)
        print("testing ", model_name)
        open("classification_{}".format(model_name), 'a').close()
        torch.manual_seed(0)
        model = models.__dict__[model_name](num_classes=50)
        scripted_model = torch.jit.script(model)
        scripted_model.eval()
        x = torch.rand(input_shape)
        py_output = model(x)
        scripted_model(x)
        opt_output = scripted_model(x)
        #assert torch.allclose(py_output, opt_output)
        plan = get_plan(scripted_model)
        num_bailouts = plan.code.num_bailouts()
        print(num_bailouts)
        for i in range(0, num_bailouts):
            plan.code.request_bailout(i)
            bailout_output = scripted_model(x)
            #assert torch.allclose(bailout_output, opt_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', nargs=1)
    parsed_args = parser.parse_args()
    do_detection_test(parsed_args.model_name[0])




