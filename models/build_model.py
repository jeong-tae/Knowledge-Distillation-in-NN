import torchvision

def build_model(model_name):
    if model_name.startswith('resnet'):
        model = torchvision.models.resnet.__dict__[model_name]
        model = model(pretrained=True)
        if '101' in model_name or '152' in args.model:
            model.fc = nn.Linear(512*4, args.num_classes)
        else:
            model.fc = nn.Linear(512, args.num_classes)
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
    elif model_name.startswith('squeezenet'):
        model = torchvision.models.squeezenet.__dict__[model_name]
        model = model(pretrained=True)
        model.classifier._modules["1"] = nn.Conv2d(512, args.num_classes, kernel_size=(1, 1))
        model.num_classes = args.num_classes
    elif model_name.startswith('vgg'):
        model = torchvision.models.vgg.__dict__[model_name]
        model = model(pretrained=True)
        model.classifier[6] = nn.Linear(4096, args.num_classes)
    elif model_name.startswith('densenet'):
        model = torchvision.models.densenet.__dict__[model_name]
        model = model(num_classes=args.num_classes, pretrained=True)
    else:
        raise NotImplementedError(" [!] Not implemented model name is given: %s, Please correct the model name or define your model on models directory"%model_name)

    return model


