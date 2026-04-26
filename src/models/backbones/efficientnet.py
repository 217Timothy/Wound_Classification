import timm


def build_efficientnet(model_name="efficientnet-b0", pretrained=True):
    model = timm.create_model(model_name, 
                              pretrained=pretrained, 
                              num_classes=0)
    return model, model.num_features