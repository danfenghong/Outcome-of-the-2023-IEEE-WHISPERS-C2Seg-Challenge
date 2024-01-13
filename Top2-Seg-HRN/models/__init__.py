from .SegHRNet import SegHRNet
from .SegHRNet_DA import SegHRNet_DA

def build_model(in_ch, 
                n_classes, 
                model_key='', 
                backbone='resnet34', 
                pretrained_flag=False, 
                resume_path=''):
    assert model_key in ['SegHRNet_DA', 'SegHRNet'], '%s not registered in models' % model_key
    
    if model_key == 'SegHRNet_DA':
        assert backbone in ['hr-w18', 'hr-w32', 'hr-w48']
        model = SegHRNet_DA(in_ch, 
                         n_classes,backbone)
        if pretrained_flag:
            model.init_weights(resume_path) 
    elif model_key == 'SegHRNet':
        assert backbone in ['hr-w18', 'hr-w32', 'hr-w48']
        model = SegHRNet(in_ch, 
                         n_classes,backbone)
        if pretrained_flag:
            model.init_weights(resume_path)
    else:
        pass
    return model