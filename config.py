class Config:

	
    VOC_ROOT = '/Users/liuhaoxin/data/VOC/VOCdevkit'

    num_classes = 11

    resume = None
    
    lr = 0.001
    batch_size = 32 
    
    momentum = 0.9
    weight_decay = 5e-4

    epoch = 100 

    gamma = 0.2
    lr_reduce_epoch = 30


    save_folder = 'weights/'
    basenet = 'vgg16_reducedfc.pth'

    log_fn = 10 

    neg_radio = 3
    
    min_size = 300
    
    grids = (38, 19, 10, 5, 3, 1)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)] 
    anchor_num = [4, 6, 6, 6, 4, 4]

    mean = (104, 117, 123)
    variance = (0.1, 0.2)



opt = Config()
