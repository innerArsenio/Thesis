import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import timm
from dataset.isic_dataset import SkinDataset
from model import ExpLICD, ExpLICD_Self
from torchvision import transforms, models
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
import copy
from torch.utils.data import DataLoader
from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import utils
import matplotlib.pyplot as plt
import os
import sys
import time
import math
import pdb
import schedulefree
DEBUG = False



dataset_dict = {
    'isic2018': SkinDataset,
}

WEAK_SUPERVISION_WEIGHT = 0

def train_net(model, config):

    print(config.unique_name)
    
    train_transforms = copy.deepcopy(config.preprocess)
    train_transforms.transforms.pop(0)
    if model.model_name != 'clip':
        train_transforms.transforms.pop(0)
    train_transforms.transforms.insert(0, transforms.RandomVerticalFlip())
    train_transforms.transforms.insert(0, transforms.RandomHorizontalFlip())
    train_transforms.transforms.insert(0, transforms.RandomResizedCrop(size=(224,224), scale=(0.75, 1.0), ratio=(0.75, 1.33), interpolation=utils.get_interpolation_mode('bicubic')))
    train_transforms.transforms.insert(0, transforms.ToPILImage())
    #if config.dataset == 'isic2018':
    #    train_transforms.transforms.insert(-1, utils.gray_world()) 


    val_transforms = copy.deepcopy(config.preprocess)
    val_transforms.transforms.insert(0, transforms.ToPILImage())
    #if config.dataset == 'isic2018':
    #    val_transforms.transforms.insert(-1, utils.gray_world())


    trainset = dataset_dict[config.dataset](config.data_path, mode='train', transforms=train_transforms, flag=config.flag, debug=DEBUG, config=config, return_concept_label=True)
    trainLoader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)

    valset = dataset_dict[config.dataset](config.data_path, mode='val', transforms=val_transforms, flag=config.flag, debug=DEBUG, config=config, return_concept_label=True)
    valLoader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)
    
    testset = dataset_dict[config.dataset](config.data_path, mode='test', transforms=val_transforms, flag=config.flag, debug=DEBUG, config=config, return_concept_label=True)
    testLoader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)


    
    writer = SummaryWriter(config.log_path+config.unique_name)
    
    if config.cls_weight == None:
        criterion = nn.CrossEntropyLoss().cuda() 
    else:
        lesion_weight = torch.FloatTensor(config.cls_weight).cuda()
        criterion = nn.CrossEntropyLoss(weight=lesion_weight).cuda()
    
    # if config.optimizer == 'sgd':
    #     optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0005)
    # elif config.optimizer == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # elif config.optimizer == 'adamw':
    #     optimizer = optim.AdamW([
    #         {'params': model.get_backbone_params(), 'lr': config.lr * 0.1},
    #         {'params': model.get_bridge_params(), 'lr': config.lr},
    #     ])

    scaler = torch.cuda.amp.GradScaler() if config.amp else None
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=config.lr, warmup_steps=5000 , weight_decay=0.1)

    BMAC, acc, f1, _, _ = validation(model, valLoader, criterion)
    print('BMAC: %.5f, Acc: %.5f, f1: %.5f'%(BMAC, acc, f1))

    best_acc = 0
    best_f1 = 0
    print_the_target = True
    for epoch in range(config.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, config.epochs))
        batch_time = 0
        epoch_loss_cls = 0
        epoch_loss_concept = 0


        model.train()
        optimizer.train()
        
        end = time.time()
        
        exp_scheduler = utils.exp_lr_scheduler_with_warmup(optimizer, init_lr=config.lr, epoch=epoch, warmup_epoch=config.warmup_epoch, max_epoch=config.epochs)

        for i, (data, label, concept_label, sincerity) in enumerate(trainLoader, 0):
            x, target = data.float().cuda(), label.long().cuda()
            concept_label = concept_label.long().cuda()
            sincerity = sincerity.long().cuda()
            if print_the_target:
                #print(f"x shape", x.shape)
                #print(f"target shape", target.shape)
                #print(f"concept_label ", concept_label)
                print_the_target = False
            
            optimizer.zero_grad()
            loss_true= torch.tensor(0.0, requires_grad=True).cuda()
            loss_synth = torch.tensor(0.0, requires_grad=True).cuda()

            image_concept_loss_true= torch.tensor(0.0, requires_grad=True).cuda()
            image_concept_loss_synth = torch.tensor(0.0, requires_grad=True).cuda()

            true_logit = [s == 1 for s in sincerity]
            synth_logit = [s == 0 for s in sincerity]
            if config.amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    cls_logits, image_logits_dict = model(x)
                   
                    loss_cls = criterion(cls_logits, target)

                    loss_concepts = 0
                    idx = 0
                    for key in net.concept_token_dict.keys():
                        image_concept_loss = F.cross_entropy(image_logits_dict[key], concept_label[:, idx])
                        loss_concepts += image_concept_loss
                        idx += 1

                    loss = loss_cls + loss_concepts / idx

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            else:
                cls_logits, image_logits_dict = model(x)

                true_output_cls = cls_logits[true_logit]
                synth_output_cls= cls_logits[synth_logit]

                if true_output_cls.size(0) > 0:
                    loss_true = criterion(true_output_cls,  target[true_logit])

                if synth_output_cls.size(0) > 0:
                    loss_synth = WEAK_SUPERVISION_WEIGHT *criterion(synth_output_cls,  target[synth_logit])

                #loss_cls = criterion(cls_logits, target)

                loss_cls =loss_true + loss_synth

                loss_concepts = 0
                idx = 0
                
                for key in net.concept_token_dict.keys():
                    #print(image_logits_dict[key].shape)
                    true_output_concept = image_logits_dict[key][true_logit]
                    synth_output_concept = image_logits_dict[key][synth_logit]
                    if true_output_concept.size(0) > 0:
                        image_concept_loss_true = F.cross_entropy(true_output_concept, concept_label[true_logit, idx])
                    if synth_output_concept.size(0) > 0:
                        image_concept_loss_synth = WEAK_SUPERVISION_WEIGHT *F.cross_entropy(synth_output_concept, concept_label[synth_logit, idx])
                    loss_concepts +=image_concept_loss_true+image_concept_loss_synth

                    # image_concept_loss=F.cross_entropy(image_logits_dict[key], concept_label[:, idx])
                    # loss_concepts += image_concept_loss
                    # print(image_logits_dict[key][0])
                    # print()
                    # print(concept_label[:, idx][0])
                    #loss_concepts +=image_concept_loss_true+image_concept_loss_synth
                    idx += 1

                loss = loss_cls + loss_concepts / idx

                loss.backward()
                optimizer.step()
            


            epoch_loss_cls +=  loss_cls.item()
            epoch_loss_concept += loss_concepts.item()

            batch_time = time.time() - end

            end = time.time()


        #     print(i, 'loss_cls: %.5f, loss_concept: %.5f, batch_time: %.5f' % (loss.item(), loss_concepts.item(), batch_time))
        
        # print('[epoch %d] epoch loss_cls: %.5f, epoch_loss_concept: %.5f' % (epoch+1, epoch_loss_cls/(i+1), epoch_loss_concept/(i+1) ))

        writer.add_scalar('Train/Loss_cls', epoch_loss_cls/(i+1), epoch+1)
        writer.add_scalar('Train/Loss_concept', epoch_loss_concept/(i+1), epoch+1)


        if not os.path.isdir('%s%s/'%(config.cp_path, config.unique_name)):
            os.makedirs('%s%s/'%(config.cp_path, config.unique_name))
        
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), '%s%s/CP%d.pth'%(config.cp_path, config.unique_name, epoch+1))

        optimizer.eval()
        val_BMAC, val_acc, val_f1, val_loss_cls, val_loss_concept = validation(model, valLoader, criterion)
        writer.add_scalar('Val/BMAC', val_BMAC, epoch+1)
        writer.add_scalar('Val/Acc', val_acc, epoch+1)
        writer.add_scalar('Val/val_loss_cls', val_loss_cls, epoch+1)
        writer.add_scalar('Val/val_loss_concept', val_loss_concept, epoch+1)
        
        test_BMAC, test_acc, test_f1, test_loss_cls, test_loss_concept = validation(model, testLoader, criterion)
        # print('Test/Acc', test_acc)
        # print('Test Balanced Acc', test_BMAC)
        
        writer.add_scalar('Test/BMAC', test_BMAC, epoch+1)
        writer.add_scalar('Test/Acc', test_acc, epoch+1)
        writer.add_scalar('Test/test_loss_cls', test_loss_cls, epoch+1)
        writer.add_scalar('Test/test_loss_concept', test_loss_concept, epoch+1)
                
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR/lr', lr, epoch+1)


        if val_BMAC >= best_acc:
            best_acc = val_BMAC
            best_f1 = val_f1
            if not os.path.exists(config.cp_path):
                os.makedirs(config.cp_path)
            torch.save(model.state_dict(), '%s%s/best.pth'%(config.cp_path, config.unique_name))
            print('Test/Acc', test_acc)
            print('Test f1', test_f1)
            print('Test Balanced Acc', test_BMAC)
          

        print('save done')
        print('BMAC: %.5f/best BMAC: %.5f, Acc: %.5f'%(val_BMAC, best_acc, val_acc))


        
def validation(model, dataloader, criterion):
    
    net = model

    net.eval()
    
    losses_cls = 0
    losses_concepts = 0

    pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    with torch.no_grad():
        for i, (data, label, concept_label, _) in enumerate(dataloader):
            
            data, label = data.cuda(), label.long().cuda()
            concept_label = concept_label.long().cuda()
            cls_logits, image_logits_dict = net(data)

            loss_cls = criterion(cls_logits, label)
            losses_cls += loss_cls.item()

            tmp_loss_concepts = 0
            idx = 0
            for key in net.concept_token_dict.keys():
                image_concept_loss = F.cross_entropy(image_logits_dict[key], concept_label[:, idx])
                tmp_loss_concepts += image_concept_loss.item()
                idx += 1

            losses_concepts += tmp_loss_concepts / len(list(net.concept_token_dict.keys()))

            _, label_pred = torch.max(cls_logits, dim=1)
            
            
            pred_list = np.concatenate((pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0)
            gt_list = np.concatenate((gt_list, label.cpu().numpy().astype(np.uint8)), axis=0)
    
    BMAC = balanced_accuracy_score(gt_list, pred_list)
    correct = np.sum(gt_list == pred_list)
    acc = 100 * correct / len(pred_list)
    f1 = f1_score(gt_list, pred_list, average='macro')

    return BMAC, acc, f1, losses_cls/(i+1), losses_concepts/(i+1)




if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=80, type='int',
            help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=128,
            type='int', help='batch size')
    parser.add_option('--warmup_epoch', dest='warmup_epoch', default=5, type='int')
    parser.add_option('--optimizer', dest='optimizer', default='adamw', type='str')
    parser.add_option('-l', '--lr', dest='lr', default=0.0001, 
            type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False,
            help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path',
            #default='/data/yunhe/Liver/auto-aug/checkpoint/', help='checkpoint path')
            default='./checkpoint/', help='checkpoint path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path', 
            default='./log/', help='log path')
    parser.add_option('-m', '--model', type='str', dest='model',
            default='explicd', help='use which model')
    parser.add_option('--linear-probe', dest='linear_probe', action='store_true', help='if use linear probe finetuning')
    parser.add_option('-d', '--dataset', type='str', dest='dataset', 
            default='isic2018', help='name of dataset')
    parser.add_option('--data-path', type='str', dest='data_path', 
            default='/data/local/yg397/dataset/isic2018/', help='the path of the dataset')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name',
            default='test', help='name prefix')
     

    parser.add_option('--flag', type='int', dest='flag', default=2)

    parser.add_option('--gpu', type='str', dest='gpu',
            default='0')
    parser.add_option('--amp', action='store_true', help='if use mixed precision training')

    (config, args) = parser.parse_args()
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    config.log_path = config.log_path + config.dataset + '/'
    config.cp_path = config.cp_path + config.dataset + '/'
    
    print('use model:', config.model)
    
    num_class_dict = {
        'isic2018': 7,
    }

    # cls_weight_dict = {
    #     'isic2018': [1, 0.5, 1.2, 1.3, 1, 2, 2], 
    # }

    cls_weight_dict = {
        'isic2018': [0.039, 0.0065 , 0.084 , 0.134, 0.039,  0.389, 0.305], 
    }
    
    config.cls_weight = cls_weight_dict[config.dataset]
    config.num_class = num_class_dict[config.dataset]

    
    from concept_dataset import explicid_isic_dict
    concept_list = explicid_isic_dict
    #net = ExpLICD(concept_list=concept_list, model_name='biomedclip', config=config)
    net =ExpLICD_Self(concept_list=concept_list, model_name='biomedclip', config=config)

    # We find using orig_in21k vit weights works better than biomedclip vit weights
    # Delete the following if want to use biomedclip weights
    vit = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True, num_classes=config.num_class)
    vit.head = nn.Identity()
    net.model.visual.trunk.load_state_dict(vit.state_dict())



    if config.load:
        net.load_state_dict(torch.load(config.load))
        print('Model loaded from {}'.format(config.load))

    net.cuda()
    

    train_net(net, config)

    print('done')
        

