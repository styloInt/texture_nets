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

# styles = list(set(styles) - set(except_them))
# [x if x.find() for]

styles = ['styles/Ashville-Willem-de-Kooning-1947.jpg'          ]

q = Queue.Queue(maxsize=3)
q.put(0)
q.put(1)
q.put(2)


def runer(fname):
    gpu = q.get()
    print fname, gpu

    # if not os.path.exists('data/checkpoints_instance/%s' % os.path.basename(fname)):
    # os.system("CUDA_VISIBLE_DEVICES=%s th train.lua -checkpoints_path data/checkpoints_instance -data ~/data/MSCOCO/dataset/ -style_image %s -style_size 600 -image_size 512 -model johnson_small -batch_size 1 -learning_rate 1e-3 -style_weight 15 -style_layers relu1_2,relu2_2,relu3_2,relu4_2 -content_layers relu4_2 -nThreads 8 -display_port 8003 -num_iterations 100000" % (gpu, fname))
    os.system("CUDA_VISIBLE_DEVICES=%s th train.lua -checkpoints_path data/checkpoints_instance_w10 -data ~/data/MSCOCO/dataset/ -style_image %s -style_size 600 -image_size 512 -model johnson_small -batch_size 1 -learning_rate 1e-3 -style_weight 10 -style_layers relu1_2,relu2_2,relu3_2,relu4_2 -content_layers relu4_2 -nThreads 8 -display_port 8003 -num_iterations 100000" % (gpu, fname))
    # else:
        # print('exists')
    q.put(gpu)

Parallel(n_jobs=3, backend="threading")(delayed(runer)(s) for s in styles)
