import argparse
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models


def do_classification_test(model_name, x):

    print("testing ", model_name)
    open("classification_{}".format(model_name), 'a').close()
    torch.manual_seed(0)
    model = models.__dict__[model_name](num_classes=1000, pretrained=True)
    py_output = model(x)
    print(py_output.size())
    return py_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--image')
    parsed_args = parser.parse_args()

    img = Image.open(parsed_args.image)

    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    x = transformations(img)
    out1 = do_classification_test(parsed_args.model, torch.unsqueeze(x,0))

    import json
    idx2label = []
    cls2label = {}
    with open("imagenet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

    print(idx2label[out1.argmax(dim = 1).item()])


