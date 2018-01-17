import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2,1,0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *
from utility.file import *

from net.rate import *
from net.loss import *

from dataset.audio_dataset import *
from dataset.audio_processing_tf import *

from itertools import compress
# -------------------------------------------------------------------------------------
from train_resnet3 import *
test_augment = valid_augment



#--------------------------------------------------------------

def do_submit():

    out_dir  = '.'
    initial_checkpoint = \
        './checkpoint/00039000_model.pth'

    #output
    csv_file          = out_dir +'/submit/submission.csv'
    csv_file_ensemble = out_dir +'/submit/submission_ensemble.csv'
    memmap_file       = out_dir +'/submit/probs.uint8.memmap'
    memmap_file_ll    = out_dir +'/submit/probs_ll.uint8.memmap'


    ## setup -----------------------------
    os.makedirs(out_dir +'/submit', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.submit.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\n')


    ## net ---------------------------------
    log.write('** net setting **\n')

    net = Net(in_shape = (1, -1, -1), num_classes=AUDIO_NUM_CLASSES).cuda()
    net.load_state_dict(torch.load(initial_checkpoint))
    net.eval()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('%s\n\n'%(type(net)))


    ## dataset ---------------------------------
    log.write('** dataset setting **\n')

    test_augment_normalized = partial(test_augment, mean=(2.0641,  -1.1881), std=(24.4161, 10.0548))

    test_dataset = AudioDataset(
                                'test_158538',  mode='test',
                                 transform = test_augment_normalized)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = 16,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = collate)
    test_num  = len(test_loader.dataset)


    ## start submission here ####################################################################
    start = timer()

    norm_probs = np.memmap(memmap_file, dtype='uint8', mode='r', shape=(test_num, AUDIO_NUM_CLASSES))
    norm_probs_ll = np.memmap(memmap_file_ll, dtype='uint8', mode='r', shape=(test_num, AUDIO_NUM_CLASSES, 9))
    names = np.zeros(test_num, np.int32)

    if False:
        n = 0
        for tensors_mel, tensors_mfcc, labels, indices in test_loader:
            print('\rpredicting: %10d/%d (%0.0f %%)  %0.2f min'%(n, test_num, 100*n/test_num,
                             (timer() - start) / 60), end='',flush=True)
            #time.sleep(0.01)

            # forward
            tensors_mel  = Variable(tensors_mel, volatile=True).cuda(async=True)
            tensors_mfcc = Variable(tensors_mfcc,volatile=True).cuda(async=True)
            #logits, l_1,l_2,l_3,l_4,l_5,l_6,  l_7,l_8,l_9,l_10,l_11,l_12  = data_parallel(net, (tensors_mel, tensors_mfcc))
            logits, l_3,l_4,l_5,l_6,  l_8,l_9,l_10,l_11,l_12  = data_parallel(net, (tensors_mel, tensors_mfcc))

            #ll = torch.stack((l_1,l_2,l_3,l_4,l_5,l_6,  l_7,l_8,l_9,l_10,l_11,l_12), 2) # batch_size, num_classes, 12
            ll = torch.stack((l_3,l_4,l_5,l_6,  l_8,l_9,l_10,l_11,l_12), 2) # batch_size, num_classes, 12

            probs    = F.softmax(logits,dim=1)
            probs_ll = F.softmax(ll, dim=1)
            labels  = probs.topk(1)[1]
            labels = labels.data.cpu().numpy().reshape(-1)
            probs  = probs.data.cpu().numpy()*255
            probs  = probs.astype(np.uint8)

            probs_ll  = probs_ll.data.cpu().numpy()*255
            probs_ll  = probs_ll.astype(np.uint8)

            batch_size = len(indices)
            names[n:n+batch_size]=labels
            norm_probs[n:n+batch_size]=probs
            norm_probs_ll[n:n+batch_size]=probs_ll
            n += batch_size

        print('\n')
        assert(n == len(test_loader.sampler) and n == test_num)


    ## submission csv  ----------------------------
    fnames = [id.split('/')[-1] for id in test_dataset.ids]
    #names  = [AUDIO_NAMES[l] for l in names]
    #df = pd.DataFrame({ 'fname' : fnames , 'label' : names})
    #df.to_csv(csv_file, index=False)

    norm_probs_ll = norm_probs_ll ** 0.5
    names_mean = list(norm_probs_ll.mean(axis=(2)).argmax(axis=1))
    names_mean = [AUDIO_NAMES[l] for l in names_mean]
    df = pd.DataFrame({ 'fname' : fnames , 'label' : names_mean})
    df.to_csv(csv_file_ensemble, index=False)

    th = 0.9 * 255

    unknown_probs = ((norm_probs[:, 1]) > th) & \
        (norm_probs_ll[:, 1, 0] > th) & \
        (norm_probs_ll[:, 1, 1] > th) & \
        (norm_probs_ll[:, 1, 2] > th) & \
        (norm_probs_ll[:, 1, 3] > th) & \
        (norm_probs_ll[:, 1, 4] > th) & \
        (norm_probs_ll[:, 1, 5] > th) & \
        (norm_probs_ll[:, 1, 6] > th) & \
        (norm_probs_ll[:, 1, 7] > th) & \
        (norm_probs_ll[:, 1, 8] > th) 

    unknowns = list(compress(fnames, unknown_probs))
    df = pd.DataFrame({ 'fname' : unknowns})
    df.to_csv('unknowns_.csv',  header=False, index=False)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_submit()


    print('\nsucess!')