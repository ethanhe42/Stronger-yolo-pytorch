def build_nework(self, input_data, val_reuse=False):
    """
    :param input_data: shape为(batch_size, input_size, input_size, 3)
    :return: conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox
    conv_sbbox的shape为(batch_size, input_size / 8, input_size / 8, gt_per_grid * (5 + num_classes))
    conv_mbbox的shape为(batch_size, input_size / 16, input_size / 16, gt_per_grid * (5 + num_classes))
    conv_lbbox的shape为(batch_size, input_size / 32, input_size / 32, gt_per_grid * (5 + num_classes))
    conv_?是YOLO的原始卷积输出(raw_dx, raw_dy, raw_dw, raw_dh, raw_conf, raw_prob)
    pred_sbbox的shape为(batch_size, input_size / 8, input_size / 8, gt_per_grid, 5 + num_classes)
    pred_mbbox的shape为(batch_size, input_size / 16, input_size / 16, gt_per_grid, 5 + num_classes)
    pred_lbbox的shape为(batch_size, input_size / 32, input_size / 32, gt_per_grid, 5 + num_classes)
    pred_?是YOLO预测bbox的信息(x, y, w, h, conf, prob)，(x, y, w, h)的大小是相对于input_size的
    """
    with tf.variable_scope('yolov3', reuse=val_reuse):
        darknet_route0, darknet_route1, darknet_route2 = darknet53(input_data, self.__training)

        conv = convolutional(name='conv0', input_data=darknet_route2, filters_shape=(1, 1, 1024, 512),
                             training=self.__training)
        conv = convolutional(name='conv1', input_data=conv, filters_shape=(3, 3, 512, 1024),
                             training=self.__training)
        conv = convolutional(name='conv2', input_data=conv, filters_shape=(1, 1, 1024, 512),
                             training=self.__training)
        conv = convolutional(name='conv3', input_data=conv, filters_shape=(3, 3, 512, 1024),
                             training=self.__training)
        conv = convolutional(name='conv4', input_data=conv, filters_shape=(1, 1, 1024, 512),
                             training=self.__training)

        # ----------**********---------- Detection branch of large object ----------**********----------
        conv_lbbox = convolutional(name='conv5', input_data=conv, filters_shape=(3, 3, 512, 1024),
                                   training=self.__training)
        conv_lbbox = convolutional(name='conv6', input_data=conv_lbbox,
                                   filters_shape=(1, 1, 1024, self.__gt_per_grid * (self.__num_classes + 5)),
                                   training=self.__training, downsample=False, activate=False, bn=False)
        pred_lbbox = decode(name='pred_lbbox', conv_output=conv_lbbox,
                            num_classes=self.__num_classes, stride=self.__strides[2])
        # ----------**********---------- Detection branch of large object ----------**********----------

        # ----------**********---------- up sample and merge features map ----------**********----------
        conv = convolutional(name='conv7', input_data=conv, filters_shape=(1, 1, 512, 256),
                             training=self.__training)
        conv = upsample(name='upsample0', input_data=conv)
        conv = route(name='route0', previous_output=darknet_route1, current_output=conv)
        # ----------**********---------- up sample and merge features map ----------**********----------

        conv = convolutional('conv8', input_data=conv, filters_shape=(1, 1, 512 + 256, 256),
                             training=self.__training)
        conv = convolutional('conv9', input_data=conv, filters_shape=(3, 3, 256, 512),
                             training=self.__training)
        conv = convolutional('conv10', input_data=conv, filters_shape=(1, 1, 512, 256),
                             training=self.__training)
        conv = convolutional('conv11', input_data=conv, filters_shape=(3, 3, 256, 512),
                             training=self.__training)
        conv = convolutional('conv12', input_data=conv, filters_shape=(1, 1, 512, 256),
                             training=self.__training)

        # ----------**********---------- Detection branch of middle object ----------**********----------
        conv_mbbox = convolutional(name='conv13', input_data=conv, filters_shape=(3, 3, 256, 512),
                                   training=self.__training)
        conv_mbbox = convolutional(name='conv14', input_data=conv_mbbox,
                                   filters_shape=(1, 1, 512, self.__gt_per_grid * (self.__num_classes + 5)),
                                   training=self.__training, downsample=False, activate=False, bn=False)
        pred_mbbox = decode(name='pred_mbbox', conv_output=conv_mbbox,
                            num_classes=self.__num_classes, stride=self.__strides[1])
        # ----------**********---------- Detection branch of middle object ----------**********----------

        # ----------**********---------- up sample and merge features map ----------**********----------
        conv = convolutional(name='conv15', input_data=conv, filters_shape=(1, 1, 256, 128),
                             training=self.__training)
        conv = upsample(name='upsample1', input_data=conv)
        conv = route(name='route1', previous_output=darknet_route0, current_output=conv)
        # ----------**********---------- up sample and merge features map ----------**********----------

        conv = convolutional(name='conv16', input_data=conv, filters_shape=(1, 1, 256 + 128, 128),
                             training=self.__training)
        conv = convolutional(name='conv17', input_data=conv, filters_shape=(3, 3, 128, 256),
                             training=self.__training)
        conv = convolutional(name='conv18', input_data=conv, filters_shape=(1, 1, 256, 128),
                             training=self.__training)
        conv = convolutional(name='conv19', input_data=conv, filters_shape=(3, 3, 128, 256),
                             training=self.__training)
        conv = convolutional(name='conv20', input_data=conv, filters_shape=(1, 1, 256, 128),
                             training=self.__training)

        # ----------**********---------- Detection branch of small object ----------**********----------
        conv_sbbox = convolutional(name='conv21', input_data=conv, filters_shape=(3, 3, 128, 256),
                                   training=self.__training)
        conv_sbbox = convolutional(name='conv22', input_data=conv_sbbox,
                                   filters_shape=(1, 1, 256, self.__gt_per_grid * (self.__num_classes + 5)),
                                   training=self.__training, downsample=False, activate=False, bn=False)
        pred_sbbox = decode(name='pred_sbbox', conv_output=conv_sbbox,
                            num_classes=self.__num_classes, stride=self.__strides[0])
        # ----------**********---------- Detection branch of small object ----------**********----------

    return conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox