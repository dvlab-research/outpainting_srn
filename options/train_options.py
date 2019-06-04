import argparse
import os
import time

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--dataset', type=str, default='paris_streetview',
                                 help='dataset of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--data_file', type=str, default='', help='the file storing training file paths')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--load_model_dir', type=str, default='', help='pretrained models are given here')
        self.parser.add_argument('--model_prefix', type=str, default='snap', help='models are saved here')
        # input/output sizes
        self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        # for setting inputs
        self.parser.add_argument('--model', type=str, default='srn')
        self.parser.add_argument('--random_crop', type=int, default=1,
                                 help='using random crop to process input image when '
                                      'the required size is smaller than the given size')
        self.parser.add_argument('--random_mask', type=int, default=1,
                                 help='using random mask')
        self.parser.add_argument('--pretrain_network', type=int, default=1)
        self.parser.add_argument('--use_cn', type=int, default=1)
        self.parser.add_argument('--feat_expansion_op', type=str, default='subpixel')

        self.parser.add_argument('--gan_loss_alpha', type=float, default=1e-3)
        self.parser.add_argument('--wgan_gp_lambda', type=float, default=10)
        self.parser.add_argument('--pretrain_l1_alpha', type=float, default=1.2)
        self.parser.add_argument('--l1_loss_alpha', type=float, default=4.2)
        self.parser.add_argument('--ae_loss_alpha', type=float, default=1.2)
        self.parser.add_argument('--mrf_alpha', type=float, default=0.05)
        self.parser.add_argument('--fa_alpha', type=float, default=0.5)
        self.parser.add_argument('--random_seed', type=bool, default=False)
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')
        self.parser.add_argument('--l1_type', type=int, default=0, help='type of l1')

        self.parser.add_argument('--train_spe', type=int, default=1000)
        self.parser.add_argument('--max_iters', type=int, default=40000)
        self.parser.add_argument('--viz_max_out', type=int, default=10)
        self.parser.add_argument('--viz_steps', type=int, default=10)

        self.parser.add_argument('--img_shapes', type=str, default='256,256,3',
                                 help='given shape parameters: h,w,c or h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='128,128',
                                 help='given mask parameters: h,w')
        self.parser.add_argument('--max_delta_shapes', type=str, default='0,0')
        self.parser.add_argument('--margins', type=str, default='0,0')
        self.parser.add_argument('--gan_type', type=str, default='contextual')

        # for generator
        self.parser.add_argument('--g_cnum', type=int, default=32,
                                 help='# of generator filters in first conv layer')
        self.parser.add_argument('--d_cnum', type=int, default=64,
                                 help='# of discriminator filters in first conv layer')
        self.parser.add_argument('--vgg19_path', type=str, default='vgg19_weights/imagenet-vgg-verydeep-19.mat')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.dataset_path = self.opt.data_file
        assert os.path.isfile(self.opt.data_file)

        assert self.opt.pretrain_network == 0 or self.opt.pretrain_network == 1
        self.opt.pretrain_network = True if self.opt.pretrain_network == 1 else False

        assert self.opt.random_crop == 0 or self.opt.random_crop == 1
        self.opt.random_crop = True if self.opt.random_crop == 1 else False

        assert self.opt.random_mask == 0 or self.opt.random_mask == 1
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.use_cn == 0 or self.opt.use_cn == 1
        self.opt.use_cn = True if self.opt.use_cn == 1 else False

        # 0: relative spatial variant loss
        # 1: confidence driven loss
        # 2: common l1
        assert 2 >= self.opt.l1_type >= 0

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(str(id))
        assert self.opt.model in ['srn', 'srn-hr']
        assert self.opt.feat_expansion_op in ['subpixel', 'deconv', 'bilinear-conv', 'unfold']
        assert self.opt.gan_type in ['contextual', 'wgan']

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.opt.gpu_ids)

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        str_max_delta_shapes = self.opt.max_delta_shapes.split(',')
        self.opt.max_delta_shapes = [int(x) for x in str_max_delta_shapes]

        str_margins = self.opt.margins.split(',')
        self.opt.margins = [int(x) for x in str_margins]

        # model name and date
        self.opt.date_str = time.strftime('%Y%m%d-%H%M%S')
        self.opt.model_folder = self.opt.date_str + '_' + self.opt.dataset + '_' + self.opt.model
        if self.opt.model == 'srn':
            self.opt.model_folder += '_' + self.opt.feat_expansion_op
            self.opt.model_folder += '_wo-cn' if self.opt.use_cn is False else ''
            self.opt.model_folder += '_adaIN' if self.opt.fa_alpha == 1.0 else ''
        self.opt.model_folder += '_' + self.opt.gan_type
        self.opt.model_folder += '_b' + str(self.opt.batch_size)
        self.opt.model_folder += '_s' + str(self.opt.img_shapes[0]) + 'x' + str(self.opt.img_shapes[1])
        self.opt.model_folder += '_gc' + str(self.opt.g_cnum)
        self.opt.model_folder += '_dc' + str(self.opt.d_cnum)
        if self.opt.l1_type > 0:
            self.opt.model_folder += '_l1-confidence' if self.opt.l1_type == 1 else '_l1-common'

        self.opt.model_folder += '_rand-mask' if self.opt.random_mask else ''
        self.opt.model_folder += '_pretrain' if self.opt.pretrain_network else ''

        if os.path.isdir(self.opt.checkpoints_dir) is False:
            os.mkdir(self.opt.checkpoints_dir)

        self.opt.model_folder = os.path.join(self.opt.checkpoints_dir, self.opt.model_folder)
        if os.path.isdir(self.opt.model_folder) is False:
            os.mkdir(self.opt.model_folder)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.opt.gpu_ids)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
