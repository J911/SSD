from config import opt
import numpy as np
from lib.model import SSD
import torch
import torch.nn.functional as F
import os
from lib.utils import detection_collate

from lib.multibox_encoder import MultiBoxEncoder
from lib.ssd_loss import MultiBoxLoss

from voc_dataset import VOCDetection
from custom_dataset import CustomDetection


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def print_config(path):
    with open(path, 'r') as f:
        for line in f.readlines():
            if '=' in line:
                print(line)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = opt.lr * (gamma ** (step))
    print('change learning rate, now learning rate is :', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':

    best_loss = 999

    print_config('config.py')
    print('now runing on device : ', device)

    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)

    model = SSD(opt.num_classes, opt.anchor_num)
    if opt.resume:
        print('loading checkpoint...')
        model.load_state_dict(torch.load(opt.resume))
    else:
        vgg_weights = torch.load(opt.save_folder + opt.basenet)
        print('Loading base network...')
        model.vgg.load_state_dict(vgg_weights)

     
    model.to(device)
    mb = MultiBoxEncoder(opt)
        
    # image_sets = [['2007', 'trainval'], ['2012', 'trainval']]
    # dataset = VOCDetection(opt, image_sets=image_sets, is_train=True)
    dataset = CustomDetection(opt, root='/DATA', dbtype='train')
    val_dataset = CustomDetection(opt, root='/DATA', dbtype='val')
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, collate_fn=detection_collate, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, collate_fn=detection_collate, num_workers=4)

    criterion = MultiBoxLoss(opt.num_classes, opt.neg_radio).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    print('start training........')
    for e in range(opt.epoch):
        if e % opt.lr_reduce_epoch == 0:
            adjust_learning_rate(optimizer, opt.gamma, e//opt.lr_reduce_epoch)
        total_loc_loss = 0
        total_cls_loss = 0
        total_loss = 0
        
        model.train()
        for i , (img, boxes) in enumerate(dataloader):
            img = img.to(device)
            gt_boxes = []
            gt_labels = []
            for box in boxes:
                labels = box[:, 4]
                box = box[:, :-1]
                match_loc, match_label = mb.encode(box, labels)
            
                gt_boxes.append(match_loc)
                gt_labels.append(match_label)
            
            gt_boxes = torch.FloatTensor(gt_boxes).to(device)
            gt_labels = torch.LongTensor(gt_labels).to(device)


            p_loc, p_label = model(img)


            loc_loss, cls_loss = criterion(p_loc, p_label, gt_boxes, gt_labels)

            loss = loc_loss + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()
            if i % opt.log_fn == 0:
                avg_loc = total_loc_loss / (i+1)
                avg_cls = total_cls_loss / (i+1)
                avg_loss = total_loss / (i+1)
                print('train:')
                print('epoch[{}] | batch_idx[{}] | loc_loss [{:.2f}] | cls_loss [{:.2f}] | avg_loss [{:.2f}]'.format(e, i, avg_loc, avg_cls, avg_loss))

        total_val_loc_loss = 0
        total_val_cls_loss = 0
        total_val_loss = 0
        model.eval()
        for i , (img, boxes) in enumerate(val_dataloader)
            img = img.to(device)
            gt_boxes = []
            gt_labels = []
            for box in boxes:
                labels = box[:, 4]
                box = box[:, :-1]
                match_loc, match_label = mb.encode(box, labels)
            
                gt_boxes.append(match_loc)
                gt_labels.append(match_label)
            
            gt_boxes = torch.FloatTensor(gt_boxes).to(device)
            gt_labels = torch.LongTensor(gt_labels).to(device)


            p_loc, p_label = model(img)


            loc_loss, cls_loss = criterion(p_loc, p_label, gt_boxes, gt_labels)

            loss = loc_loss + cls_loss

            optimizer.zero_grad()

            total_val_loc_loss += loc_loss.item()
            total_val_cls_loss += cls_loss.item()
            total_val_loss += loss.item()

            if i % opt.log_fn == 0:
                avg_loc = total_val_loc_loss / (i+1)
                avg_cls = total_val_cls_loss / (i+1)
                avg_loss = total_val_loss / (i+1)
                print('val:')
                print('epoch[{}] | batch_idx[{}] | loc_loss [{:.2f}] | cls_loss [{:.2f}] | avg_loss [{:.2f}]'.format(e, i, avg_loc, avg_cls, avg_loss))

            if best_loss > total_val_loss:
                best_loss = total_val_loss
                torch.save(model.state_dict(), os.path.join(opt.save_folder, 'loss-{:.2f}.pth'.format(total_loss)))

            print('best loss: ', best_loss)