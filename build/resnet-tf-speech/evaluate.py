import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2,1,0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *
from utility.file import *

from net.rate import *
from net.loss import *

from dataset.audio_dataset import *
from dataset.audio_processing_tf import *



# -------------------------------------------------------------
from train_resnet3 import *
test_augment = valid_augment



#--------------------------------------------------------------
def run_evaluate():

    out_dir  = '.'
    initial_checkpoint = \
        './checkpoint/00020000_model.pth'




    ## setup  ---------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ------------------------------
    log.write('** net setting **\n')
    net = Net(in_shape = (1, 40, 101), num_classes=AUDIO_NUM_CLASSES).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


    log.write('%s\n\n'%(type(net)))
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    test_dataset = AudioDataset(
                                'train_test_6835',  mode='train',
                                #'train_valid_6798',  mode='train',
                                #'debug_0', mode='train',
                                transform = test_augment
                                )
    test_loader  = DataLoader(
                        test_dataset,
                        sampler = SequentialSampler(test_dataset),
                        #sampler = TFSequentialSampler(test_dataset,0.1,0.1),
                        batch_size  = 16,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn=collate)


    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
    log.write('\n')


    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')
    net.eval()

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for i, (tensors, labels, indices) in enumerate(test_loader, 0):
        #print(i,test_num,indices)
        tensors = Variable(tensors,volatile=True).cuda()
        labels  = Variable(labels).cuda()

        logits = data_parallel(net, tensors)
        probs  = F.softmax(logits, dim=1)
        loss   = F.cross_entropy(logits, labels)
        acc    = top_accuracy(probs, labels, top_k=(1,))#1,5

        batch_size = len(indices)
        test_acc  += batch_size*acc[0][0]
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num


    log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
    log.write('test_acc  = %0.5f\n'%(test_acc))
    log.write('test_loss = %0.5f\n'%(test_loss))
    log.write('test_num  = %d\n'%(test_num))
    log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_evaluate()


    print('\nsucess!')