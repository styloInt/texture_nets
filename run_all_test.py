import glob
from joblib import Parallel, delayed
import Queue
import time
import os
import os.path

# except_them = ['styles/kandinsky.jpg', 'styles/prisma_4.jpg', 'styles/mondrian3.jpg','styles/color.jpg']
styles = glob.glob('styles/*')
contents = glob.glob('contents/*')
# styles = list(set(styles) - set(except_them))
# [x if x.find() for]

q = Queue.Queue(maxsize=4)
q.put(0)
q.put(2)
q.put(0)
q.put(2)
# q.put(1)
# q.put(1)


def runer(fname):
    gpu = q.get()
    print fname, gpu
    for c in contents:
        for mode in ['true', 'false']:
            for im_size in ['512', '1024']:
                os.system('mkdir -p data/out/%s' % fname)

                chkp_name = 'data/checkpoints_/%s/johnson_small/true/model_20000.t7' % os.path.basename(
                    fname)

                for trial in range(3):
                    save_path = 'data/out/%s/%s_size_%s_eval_%s_trial_%s.png' % (
                        fname, os.path.basename(c), im_size, mode, str(trial))

                    if not os.path.exists(save_path) and os.path.exists(chkp_name):
                        cmd = "CUDA_VISIBLE_DEVICES=%s th test.lua -save_path %s -model_t7 %s -input_image %s -image_size %s -eval %s" % (
                            gpu, save_path, chkp_name, c, im_size, mode)
                        os.system(cmd)
                    else:
                        print('exists')
                    # print(cmd)

    q.put(gpu)

Parallel(n_jobs=4, backend="threading")(delayed(runer)(s) for s in styles)
