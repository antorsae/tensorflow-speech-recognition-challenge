import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2,1,0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *
from utility.file import *

from net.rate import *
from net.loss import *

from dataset.audio_dataset import *
from dataset.audio_processing_tf import *
from torch_extras import one_hot 
from multiprocessing import cpu_count
from functools import partial


# -------------------------------------------------------------------------------------
from net.model.se_resnet3 import Ensemble  as Net

def train_augment(wave,label,index, mean=(0.,0.), std=(1.,1.)):

    wave = tf_random_time_shift_transform(wave, shift_limit=0.2, u=0.5)
    wave = tf_random_add_noise_transform (wave, noise_limit=0.2, u=0.5)
    wave = tf_random_pad_transform(wave)
    
    if label == 0: # silence
        if random.random() < 0.5:
            wave = np.flip(wave, axis=0)
    
    if label == 1:
        wave = librosa.effects.pitch_shift(wave, sr=AUDIO_SR, n_steps=np.random.randint(-1,2))

    #tensor = tf_wave_to_mfcc(wave)[np.newaxis,:]
    tensor_mel, tensor_mfcc = tf_wave_to_melspectrogram_mfcc(wave)

    tensor_mel -= mean[0]
    tensor_mel /= std[0]

    tensor_mfcc -= mean[1]
    tensor_mfcc /= std[1]

    tensor_mel, tensor_mfcc = torch.from_numpy(tensor_mel), torch.from_numpy(tensor_mfcc)
    return tensor_mel, tensor_mfcc, label, index

def valid_augment(wave,label,index, mean=(0.,0.), std=(1.,1.)):
    wave = tf_fix_pad_transform(wave)

    tensor_mel, tensor_mfcc = tf_wave_to_melspectrogram_mfcc(wave)
    tensor_mel -= mean[0]
    tensor_mel /= std[0]

    tensor_mfcc -= mean[1]
    tensor_mfcc /= std[1]

    tensor_mel, tensor_mfcc = torch.from_numpy(tensor_mel), torch.from_numpy(tensor_mfcc)
    return tensor_mel, tensor_mfcc, label, index

#--------------------------------------------------------------
def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (tensors_mel, tensors_mfcc, labels, indices) in enumerate(test_loader, 0):

        tensors_mel  = Variable(tensors_mel,volatile=True).cuda()
        tensors_mfcc = Variable(tensors_mfcc,volatile=True).cuda()

        batch_size = len(indices)
        nb_classes = 12
        #labels_onehot = Variable(one_hot((batch_size, nb_classes), labels.view(-1, 1)).float()).cuda()
        labels  = Variable(labels).cuda()
        logits, l_3,l_4,l_5,l_6,  l_8,l_9,l_10,l_11,l_12 = data_parallel(net, (tensors_mel, tensors_mfcc))
        probs  = F.softmax(logits, dim=1)
#        loss   = F.l1_loss(probs, labels_onehot)
        loss    = F.cross_entropy(logits, labels) 
        acc    = top_accuracy(probs, labels, top_k=(1,))#1,5

        test_acc  += batch_size*acc[0][0]
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc

def compute_stats():
    train_dataset = AudioDataset(
                                #'train_trainvalid_57886', mode='train',
                                'train_train', unknowns_csv='unknowns_train.csv', mode='train',
                                transform = train_augment)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = TFRandomSampler(train_dataset,1/12.,1/12.),
                        batch_size  = 64,
                        drop_last   = True,
                        num_workers = cpu_count()-1,
                        pin_memory  = True,
                        collate_fn  = collate)

    mel_acc_mean  = torch.zeros(1)
    mel_acc_std   = torch.zeros(1)

    mfcc_acc_mean = torch.zeros(1)
    mfcc_acc_std  = torch.zeros(1)

    it = 0.
    for tensors_mel, tensors_mfcc, labels, indices in train_loader:
        mel_acc_mean  += tensors_mel.mean()
        mel_acc_std   += tensors_mel.std()

        mfcc_acc_mean += tensors_mfcc.mean()
        mfcc_acc_std  += tensors_mfcc.std()

        it += 1.

    mel_mean  = mel_acc_mean / it
    mel_std   = mel_acc_std  / it

    mfcc_mean = mfcc_acc_mean / it
    mfcc_std  = mfcc_acc_std  / it

    print("mel_mean:",  mel_mean)
    print("mel_std:",   mel_std)

    print("mfcc_mean:", mfcc_mean)
    print("mfcc_std:",  mfcc_std)

    return mel_mean, mel_std, mfcc_mean, mfcc_std


#--------------------------------------------------------------
def run_train():

    out_dir  = '.' # s_xx1'
    initial_checkpoint = \
        None #'checkpoint/00010000_model.pth'
        # use None if train form stratch

    pretrain_file = None
    ## setup  -----------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ----------------------
    log.write('** net setting **\n')
    net = Net(in_shape = (1,40,101), num_classes=AUDIO_NUM_CLASSES).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    elif pretrain_file is not None:
        log.write('\tpretrained_file = %s\n' % pretrain_file)
        #load_pretrain_file(net, pretrain_file)


    log.write('%s\n\n'%(type(net)))
    log.write('\n')


    ## optimiser ----------------------------------
    iter_accum  = 1
    batch_size  = 96  ##NUM_CUDA_DEVICES*512 #256//iter_accum #512 #2*288//iter_accum

    num_iters   = 1000  *1000
    iter_smooth = 20
    iter_log    = 500
    iter_valid  = 500
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,1000))#1*1000

    lr_base = 0.01
    lr_factor = 5
    LR = StepLR([ (0, lr_base),  (10 * 1000, lr_base/lr_factor),  (20 * 1000, lr_base/(lr_factor**2)), (40 * 1000, lr_base/(lr_factor**3))])
    #LR = None
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.5/iter_accum, momentum=0.9, weight_decay=0.0001)

    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    mel_mean, mel_std, mfcc_mean, mfcc_std = 2.0641,  24.4161, -1.1881,  10.0548 # compute_stats()

    train_augment_normalized = partial(train_augment, mean=(mel_mean, mfcc_mean), std=(mel_std, mfcc_std))
    valid_augment_normalized = partial(valid_augment, mean=(mel_mean, mfcc_mean), std=(mel_std, mfcc_std))

    train_dataset = AudioDataset(
                                #'train_trainvalid_57886', mode='train',
                                'train_train', unknowns_csv='unknowns_train.csv', mode='train',
                                transform = train_augment_normalized)

    train_loader  = DataLoader(
                        train_dataset,
                        sampler = TFRandomSampler(train_dataset,1/12.,1/12.),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = cpu_count()-1,
                        pin_memory  = True,
                        collate_fn  = collate)

    valid_dataset = AudioDataset(
                                'train_valid', unknowns_csv='unknowns_valid.csv', mode='train',
                                 transform = valid_augment_normalized)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = TFSequentialSampler(valid_dataset, silence_probability=0.06, unknown_probability=0.5),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = cpu_count()-1,
                        pin_memory  = True,
                        collate_fn  = collate)

    log.write('\ttrain_dataset.split = %s\n'%(train_dataset.split))
    log.write('\tvalid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n'%(len(valid_loader)))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\titer_accum  = %d\n'%(iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n'%(batch_size*iter_accum))
    log.write('\n')

    #log.write(inspect.getsource(train_augment)+'\n')
    #log.write(inspect.getsource(valid_augment)+'\n')
    #log.write('\n')



    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' momentum=%f\n'% optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n'%str(LR) )

    log.write(' waves_per_epoch = %d\n\n'%len(train_dataset))
    log.write(' rate   iter_k   epoch  num_m | valid_loss/acc | train_loss/acc | batch_loss/acc |  time    \n')
    log.write('--------------------------------------------------------------------------------------------\n')


    train_loss  = 0.0
    train_acc   = 0.0
    valid_loss  = 0.0
    valid_acc   = 0.0
    batch_loss  = 0.0
    batch_acc   = 0.0
    rate = 0

    start = timer()
    j = 0
    i = 0


    while  i<num_iters:  # loop over the dataset multiple times
        sum_train_loss = 0.0
        sum_train_acc  = 0.0
        sum = 0

        net.train()
        optimizer.zero_grad()
        for tensors_mel, tensors_mfcc, labels, indices in train_loader:
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)

            if i % iter_valid==0:
                net.eval()
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.train()

                print('\r',end='',flush=True)
                log.write('%0.4f  %5.1f k  %6.2f  %4.1f | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %s \n' % \
                        (rate, i/1000, epoch, num_products/1000000, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc, \
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))



            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)[0]*iter_accum


            # one iteration update  -------------
            tensors_mel  = Variable(tensors_mel).cuda()
            tensors_mfcc = Variable(tensors_mfcc).cuda()
            batch_size = len(indices)
            nb_classes = 12
            #labels_onehot = Variable(one_hot((batch_size, nb_classes), labels.view(-1, 1)).float()).cuda()
            #logits, l_1,l_2,l_3,l_4,l_5,l_6,  l_7,l_8,l_9,l_10,l_11,l_12   = data_parallel(net, (tensors_mel, tensors_mfcc))
            logits, l_3,l_4,l_5,l_6, l_8,l_9,l_10,l_11,l_12   = data_parallel(net, (tensors_mel, tensors_mfcc))
            probs   = F.softmax(logits,dim=1)

            labels = Variable(labels).cuda()
            partial_loss = (\
                F.cross_entropy(l_3, labels) + \
                F.cross_entropy(l_4, labels) + \
                F.cross_entropy(l_5, labels) + \
                F.cross_entropy(l_6, labels) + \
                F.cross_entropy(l_8, labels) + \
                F.cross_entropy(l_9, labels) + \
                F.cross_entropy(l_10, labels) + \
                F.cross_entropy(l_11, labels) + \
                F.cross_entropy(l_12, labels) ) / 9.

            iter_cutoff = 10000
            partial_loss_factor = float(np.clip((iter_cutoff - i)/iter_cutoff, 0.1, 1.))
            cross_entropy_loss_factor = 1. - partial_loss_factor

            loss    = cross_entropy_loss_factor * F.cross_entropy(logits, labels) + partial_loss_factor * partial_loss 
            acc     = top_accuracy(probs, labels, top_k=(1,))

            # accumulated update
            loss.backward()
            if j%iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()


            # print statistics  ------------
            batch_acc  = acc[0][0]
            batch_loss = loss.data[0]
            sum_train_loss += batch_loss
            sum_train_acc  += batch_acc
            sum += 1
            if i%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                train_acc  = sum_train_acc /sum
                sum_train_loss = 0.
                sum_train_acc  = 0.
                sum = 0

            print('\r%0.4f  %5.1f k  %6.2f  %4.1f | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %s  %d,%d, %s' % \
                    (rate, i/1000, epoch, num_products/1000000, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc,
                     time_to_str((timer() - start)/60) ,i,j, str(tensors_mel.size())), end='',flush=True)
            j=j+1
        pass  #-- end of one data loader --
    pass #-- end of all iterations --


    if 1:
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()


    print('\nsucess!')
