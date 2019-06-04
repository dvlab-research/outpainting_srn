import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import add_arg_scope
import numpy as np
from functools import partial

from net.ops import random_sqaure, Margin, fixed_bbox_withMargin, bbox2mask
from net.ops import confidence_driven_mask, relative_spatial_variant_mask, deconv_frac_strided
from net.ops import flatten, gan_wgan_loss, gradients_penalty, random_interpolates
from net.ops import subpixel_conv, bilinear_conv, context_normalization, max_downsampling, unfold_conv
from net.ops import id_mrf_reg
from util.util import f2uint


class SemanticRegenerationNet:
    def __init__(self):
        self.name = 'SemanticRegenerationNet'
        self.conv5 = partial(tf.layers.conv2d, kernel_size=5, activation=tf.nn.elu, padding='SAME')
        self.conv3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        self.d_unit = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')

    @add_arg_scope
    def _deconv(self, x, filters, name='deconv', reuse=False):
        h, w = x.get_shape().as_list()[1:3]
        x = tf.image.resize_nearest_neighbor(x, [h * 2, w * 2], align_corners=True)
        with tf.variable_scope(name, reuse=reuse):
            x = self.conv3(inputs=x, filters=filters, strides=1, name=name+'_conv')
        return x

    @add_arg_scope
    def FEN(self, x, cnum):
        conv3, conv5, deconv = self.conv3, self.conv5, self._deconv
        x = conv5(inputs=x, filters=cnum, strides=1, name='conv1')
        x = conv3(inputs=x, filters=cnum * 2, strides=2, name='conv2_downsample')
        x = conv3(inputs=x, filters=cnum * 2, strides=1, name='conv3')
        x = conv3(inputs=x, filters=cnum * 4, strides=2, name='conv4_downsample')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, name='conv5')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, name='conv6')

        x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=2, name='conv7_atrous')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=4, name='conv8_atrous')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=8, name='conv9_atrous')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=16, name='conv10_atrous')

        x = conv3(inputs=x, filters=cnum * 4, strides=1, name='conv11')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, name='conv12')
        x = deconv(x, filters=cnum * 2, name='conv13_upsample')
        x = conv3(inputs=x, filters=cnum * 2, strides=1, name='conv14')
        x = deconv(x, filters=cnum, name='conv15_upsample')
        return x

    @add_arg_scope
    def CPN(self, x_fe, x_in, mask, cnum, use_cn=True, alpha=0.5):
        conv3, conv5, deconv = self.conv3, self.conv5, self._deconv
        ones_x = tf.ones_like(x_in)[:, :, :, 0:1]
        xnow = tf.concat([x_fe, x_in, mask * ones_x], axis=3)

        x = conv5(inputs=xnow, filters=cnum, strides=1, name='xconv1')
        x = conv3(inputs=x, filters=cnum, strides=2, name='xconv2_downsample')
        x = conv3(inputs=x, filters=cnum * 2, strides=1, name='xconv3')
        x = conv3(inputs=x, filters=cnum * 2, strides=2, name='xconv4_downsample')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, name='xconv5')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, name='xconv6')

        x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=2, name='xconv7_atrous')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=4, name='xconv8_atrous')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=8, name='xconv9_atrous')
        x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=16, name='xconv10_atrous')

        x = conv3(inputs=x, filters=cnum * 4, strides=1, name='allconv11')
        if use_cn:
            x = context_normalization(x, mask, alpha=alpha)
        x = conv3(inputs=x, filters=cnum * 4, strides=1, name='allconv12')
        x = deconv(x, filters=cnum * 2, name='allconv13_upsample')
        x = conv3(inputs=x, filters=cnum * 2, strides=1, name='allconv14')
        x = deconv(x, filters=cnum, name='allconv15_upsample')
        x = conv3(inputs=x, filters=cnum // 2, strides=1, name='allconv16')
        x = tf.layers.conv2d(inputs=x, kernel_size=3, filters=3, strides=1, activation=None, padding='SAME',
                             name='allconv17')
        x = tf.clip_by_value(x, -1, 1)
        return x

    def build_generator(self, x, mask, margin, config=None, reuse=False, name='inpaint_net'):
        feature_expansion_op = None
        if config is not None:
            use_cn = config.use_cn
            assert config.feat_expansion_op in ['subpixel', 'deconv', 'bilinear-conv', 'unfold']
            if config.feat_expansion_op == 'subpixel':
                feature_expansion_op = subpixel_conv
            elif config.feat_expansion_op == 'deconv':
                feature_expansion_op = deconv_frac_strided
            elif config.feat_expansion_op == 'unfold':
                feature_expansion_op = unfold_conv
            else:
                feature_expansion_op = bilinear_conv
        else:
            use_cn = True
            feature_expansion_op = subpixel_conv

        target_shape = mask.get_shape().as_list()[1:3]
        xin_expanded = tf.pad(x, [[0, 0], [margin.top, margin.bottom], [margin.left, margin.right], [0, 0]])
        xin_expanded.set_shape((x.get_shape().as_list()[0], target_shape[0], target_shape[1], 3))
        expand_scale_ratio = int(np.prod(mask.get_shape().as_list()[1:3])/np.prod(x.get_shape().as_list()[1:3]))

        # two stage network
        cnum = config.g_cnum
        with tf.variable_scope(name, reuse=reuse):
            x = self.FEN(x, cnum)
            # subpixel module, ensure the output channel the same as the input
            if config.feat_expansion_op == 'subpixel':
                x_fe = feature_expansion_op(x, cnum * expand_scale_ratio, 3, target_shape,
                                            name='feat_expansion_'+config.feat_expansion_op)
            elif config.feat_expansion_op == 'unfold':
                x_fe = feature_expansion_op(x, cnum, 3, margin, target_shape,
                                            name='feat_expansion_'+config.feat_expansion_op)
            else:
                x_fe = feature_expansion_op(x, cnum, 3, target_shape,
                                            name='feat_expansion_'+config.feat_expansion_op)

            x = self.CPN(x_fe, xin_expanded, mask, cnum, use_cn, config.fa_alpha)
        return x, x_fe

    def build_wgan_contextual_discriminator(self, x, mask, config, reuse=False):
        cnum = config.d_cnum
        dis_conv = self.d_unit
        with tf.variable_scope('D_context', reuse=reuse):
            h, w = x.get_shape().as_list()[1:3]
            x = dis_conv(x, cnum, name='dconv1')
            x = dis_conv(x, cnum*2, name='dconv2')
            x = dis_conv(x, cnum*4, name='dconv3')
            x = tf.layers.conv2d(inputs=x, kernel_size=3, filters=1, strides=1, activation=None, padding='SAME',
                                 name='dconv4')
            mask = max_downsampling(mask, ratio=8)
            x = x * mask
            x = tf.reduce_sum(x, axis=[1, 2, 3]) / tf.reduce_sum(mask, axis=[1, 2, 3])
            mask_local = tf.image.resize_nearest_neighbor(mask, [h, w], align_corners=True)
            return x, mask_local

    def build_wgan_global_discriminator(self, x, config, reuse=False):
        cnum = config.d_cnum
        dis_conv = self.d_unit
        with tf.variable_scope('D_global', reuse=reuse):
            x = dis_conv(x, cnum, name='conv1')
            x = dis_conv(x, cnum*2, name='conv2')
            x = dis_conv(x, cnum*4, name='conv3')
            x = dis_conv(x, cnum*2, name='conv4')
            x = flatten(x, name='flatten')
            return x

    def build_wgan_discriminator(self, batch_global, config, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.build_wgan_global_discriminator(
                batch_global, config=config, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_global

    def build_contextual_wgan_discriminator(self, batch_global, mask, config, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.build_wgan_global_discriminator(batch_global, config=config, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local, mask_local = self.build_wgan_contextual_discriminator(batch_global, mask,
                                                                              config=config, reuse=reuse)
            return dout_local, dout_global, mask_local

    def build_net(self, batch_data, config, summary=True, reuse=False):
        batch_pos = batch_data / 127.5 - 1.
        if config.random_mask is True:
            self.bbox_gen = random_sqaure
        else:
            self.bbox_gen = fixed_bbox_withMargin

        bbox, margin = self.bbox_gen(config)

        mask = bbox2mask(bbox, config)
        mask = 1. - mask # we need to predict context

        h, w = batch_pos.get_shape().as_list()[1:3]

        if config.random_mask is False:
            if config.mask_shapes[0] == 64: # for clothes dataset
                batch_incomplete = tf.image.crop_to_bounding_box(batch_pos,
                                                                 margin.top, margin.left,
                                                                 config.mask_shapes[0], config.mask_shapes[1])
            else:
                batch_incomplete = tf.image.crop_to_bounding_box(batch_pos, margin.top, margin.left, h, w//2)
        else:
            batch_incomplete = tf.image.crop_to_bounding_box(batch_pos, margin.top,
                                                             margin.left, config.mask_shapes[0], config.mask_shapes[1])

        if config.l1_type == 0:
            mask_priority = relative_spatial_variant_mask(mask)
        elif config.l1_type == 1:
            mask_priority = confidence_driven_mask(mask)
        else:
            mask_priority = mask
        x, x_fe = self.build_generator(batch_incomplete, mask, margin, config=config, reuse=reuse)
        batch_predicted = x

        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_pos*(1.-mask)

        if not config.pretrain_network:
            config.feat_style_layers = {'conv3_2': 1.0, 'conv4_2': 1.0}
            config.feat_content_layers = {'conv4_2': 1.0}

            config.mrf_style_w = 1.0
            config.mrf_content_w = 1.0

            losses['id_mrf_loss'] = id_mrf_reg(batch_predicted, batch_pos, config)
            tf.summary.scalar('losses/id_mrf_loss', losses['id_mrf_loss'])

        losses['l1_loss'] = config.pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x) * mask_priority)
        losses['ae_loss'] = config.pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x) * (1.-mask))
        losses['ae_loss'] /= tf.reduce_mean(1.-mask)
        if summary:
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])
            if config.random_mask is True:
                batch_incomplete_pad = tf.pad(batch_incomplete,
                                              [[0, 0], [margin.top, margin.bottom], [margin.left, margin.right], [0, 0]])
            else:
                if config.mask_shapes[0] == 64:
                    batch_incomplete_pad = tf.pad(batch_incomplete,
                                                  [[0, 0], [margin.top, margin.bottom], [margin.left, margin.right],
                                                   [0, 0]])
                else:
                    batch_incomplete_pad = batch_incomplete
            viz_img = tf.concat([batch_pos, batch_incomplete_pad, batch_complete], axis=2)[:, :, :, ::-1]
            tf.summary.image('gt__input w padding__prediction', f2uint(viz_img))

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

        # wgan with gradient penalty
        build_critics = self.build_contextual_wgan_discriminator
        # seperate gan
        global_wgan_loss_alpha = 1.0
        pos_neg_local, pos_neg_global, mask_local = build_critics(batch_pos_neg, mask, config=config, reuse=reuse)
        pos_local, neg_local = tf.split(pos_neg_local, 2)
        pos_global, neg_global = tf.split(pos_neg_global, 2)
        # wgan loss
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
        losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
        losses['d_loss'] = d_loss_global + d_loss_local
        # gp
        interpolates_global = random_interpolates(batch_pos, batch_complete)
        interpolates_local = interpolates_global
        dout_local, dout_global, _ = build_critics(interpolates_global, mask, config=config, reuse=True)
        # apply penalty
        penalty_local = gradients_penalty(interpolates_local, dout_local, mask=mask_local)
        penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
        losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local + penalty_global)
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        if summary and not config.pretrain_network:
            tf.summary.scalar('discriminator/d_loss', losses['d_loss'])
            tf.summary.scalar('wgan_loss/gp_loss', losses['gp_loss'])

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
            losses['g_loss'] += config.mrf_alpha * losses['id_mrf_loss']
        losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']
        losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def evaluate(self, images, masks, margin, config, reuse=False):
        masks = 1 - masks
        batch_pos = images / 127.5 - 1.

        h, w = batch_pos.get_shape().as_list()[1:3]

        if config.random_mask is False:
            if w == 128:
                batch_incomplete = tf.image.crop_to_bounding_box(batch_pos, 0, 0, 64, 128)
                margin = Margin(0, 0, 256-64, 0)
            elif w == 512 or w == 1024:
                batch_incomplete = tf.image.crop_to_bounding_box(batch_pos, 0, w // 4, h, w // 2)
                margin = Margin(0, w//4, 0, w//4)
            else:
                batch_incomplete = tf.image.crop_to_bounding_box(batch_pos, h // 4, w // 4, h // 2, w // 2)
                margin = Margin(h // 4, w // 4, h // 4, w // 4)
        else:
            batch_incomplete = tf.image.crop_to_bounding_box(batch_pos, margin.top, margin.left,
                                                             config.mask_shapes[0], config.mask_shapes[1])
        x, x_fe = self.build_generator(batch_incomplete, masks, margin, config=config, reuse=reuse)
        batch_predict = x
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_pos*(1-masks)
        return batch_complete, x_fe


class HRSemanticRegenerationNet(SemanticRegenerationNet):
    def __init__(self):
        super(HRSemanticRegenerationNet, self).__init__()
        self.name = 'HRSemanticRegenerationNet'

    def build_generator(self, x, mask, config=None, reuse=False, name='inpaint_net'):
        xin = x
        if config is not None:
            use_cn = config.use_cn
        else:
            use_cn = True
        # two stage network
        cnum = config.g_cnum
        with tf.variable_scope(name, reuse=reuse):
            x_fe = self.FEN(x, cnum)
            x = self.CPN(x_fe, xin, mask, cnum, use_cn, config.fa_alpha)
        return x, x_fe
    
    def build_net(self, batch_data, config, summary=True, reuse=False):
        batch_pos = batch_data / 127.5 - 1.
        if config.random_mask is True:
            self.bbox_gen = random_sqaure
        else:
            self.bbox_gen = fixed_bbox_withMargin

        bbox, _ = self.bbox_gen(config)

        mask = bbox2mask(bbox, config)
        mask = 1. - mask # we need to predict context

        h, w = batch_pos.get_shape().as_list()[1:3]

        batch_incomplete = batch_pos * (1-mask)
        
        if config.l1_type == 0:
            mask_priority = relative_spatial_variant_mask(mask)
        elif config.l1_type == 1:
            mask_priority = confidence_driven_mask(mask)
        else:
            mask_priority = mask
        x, x_fe = self.build_generator(batch_incomplete, mask, config=config, reuse=reuse)
        batch_predicted = x

        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_pos*(1.-mask)

        if not config.pretrain_network:
            config.feat_style_layers = {'conv3_2': 1.0, 'conv4_2': 1.0}
            config.feat_content_layers = {'conv4_2': 1.0}

            config.mrf_style_w = 1.0
            config.mrf_content_w = 1.0

            losses['id_mrf_loss'] = id_mrf_reg(batch_predicted, batch_pos, config)
            tf.summary.scalar('losses/id_mrf_loss', losses['id_mrf_loss'])

        losses['l1_loss'] = config.pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x) * mask_priority)
        losses['ae_loss'] = config.pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x) * (1.-mask))
        losses['ae_loss'] /= tf.reduce_mean(1.-mask)
        if summary:
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])

            viz_img = tf.concat([batch_pos, batch_incomplete, batch_complete], axis=2)[:, :, :, ::-1]
            tf.summary.image('gt__input w padding__prediction', f2uint(viz_img))

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

        # wgan with gradient penalty
        build_critics = self.build_contextual_wgan_discriminator
        # seperate gan
        global_wgan_loss_alpha = 1.0
        pos_neg_local, pos_neg_global, mask_local = build_critics(batch_pos_neg, mask, config=config, reuse=reuse)
        pos_local, neg_local = tf.split(pos_neg_local, 2)
        pos_global, neg_global = tf.split(pos_neg_global, 2)
        # wgan loss
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
        losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
        losses['d_loss'] = d_loss_global + d_loss_local
        # gp
        interpolates_global = random_interpolates(batch_pos, batch_complete)
        interpolates_local = interpolates_global
        dout_local, dout_global, _ = build_critics(interpolates_global, mask, config=config, reuse=True)
        # apply penalty
        penalty_local = gradients_penalty(interpolates_local, dout_local, mask=mask_local)
        penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
        losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local + penalty_global)
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        if summary and not config.pretrain_network:
            tf.summary.scalar('discriminator/d_loss', losses['d_loss'])
            tf.summary.scalar('wgan_loss/gp_loss', losses['gp_loss'])

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
            losses['g_loss'] += config.mrf_alpha * losses['id_mrf_loss']
        losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']
        losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def evaluate(self, images, masks, margin, config, reuse=False):
        masks = 1 - masks
        batch_pos = images / 127.5 - 1.
        batch_incomplete = batch_pos * (1 - masks)
        x, x_fe = self.build_generator(batch_incomplete, masks, config=config, reuse=reuse)
        batch_predict = x
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_pos*(1-masks)
        return batch_complete, x_fe
