import glob
from joblib import Parallel, delayed
import Queue
import time
import os
import os.path

# except_them = ['styles/kandinsky.jpg', 'styles/prisma_4.jpg', 'styles/mondrian3.jpg','styles/color.jpg']
# except_them = ['styles/fire2.jpg', 'styles/prisma_4.jpg',
#                'styles/prisma_5.jpg', 'styles/molnia2.jpg']
# styles = glob.glob('styles/*')
styles = ['styles/prisma_5.jpg', 'styles/Ashville-Willem-de-Kooning-1947.jpg', 'styles/56cc22a958742.jpg']
# styles = list(set(styles) - set(except_them))
# [x if x.find() for]

q = Queue.Queue(maxsize=3)
q.put(0)
q.put(1)
q.put(2)


def runer(fname):
    gpu = q.get()
    print fname, gpu

    if not os.path.exists('data/checkpoints/%s' % os.path.basename(fname)):
        os.system("CUDA_VISIBLE_DEVICES=%s th train.lua -data ~/data/MSCOCO/dataset/ -style_image %s -style_size 600 -image_size 512 -model johnson_small_down -batch_size 4 -learning_rate 1e-3 -style_weight 20 -style_layers relu1_2,relu2_2,relu3_2,relu4_2 -content_layers relu4_2 -nThreads 8 -pairwise_loss true -display_port 8003 -add_noise -pairwise_weight 5000 -num_iterations 20000" % (gpu, fname))
    else:
        print('exists')
    q.put(gpu)

Parallel(n_jobs=3, backend="threading")(delayed(runer)(s) for s in styles)
