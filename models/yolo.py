from .darknet import darknet19
from .utils import conv2d_bn_leaky

import tensorflow as tf
import numpy as np

import os
from typing import Tuple, List

YOLO_ANCHORS = np.array([
    [1.08, 1.19],
    [3.42, 4.41],
    [6.63, 11.38],
    [9.42, 5.11],
    [16.62, 10.52]], dtype='float32')


class YOLO():
    def __init__(self, image_shape: Tuple[int, int, int],
                 classes: List[str],
                 anchors: np.ndarray = YOLO_ANCHORS):

        height, width, _ = image_shape
        assert height % 32 == 0 and width % 32 == 0, 'Image size in YOLO_v2 must be multiples of 32.'

        self.image_shape = image_shape
        self.anchors = anchors
        self.classes = classes

        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_op = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

        self._build_model()
        self._build_loss()
        self._build_train_op()
        self._init_saver()

    @property
    def n_anchors(self):
        return len(self.anchors)

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def conv_height(self):
        return self.image_shape[0] // 32

    @property
    def conv_width(self):
        return self.image_shape[1] // 32

    @property
    def detectors_mask_shape(self):
        w, h, _ = self.image_shape
        return (w / 32, h / 32, self.n_anchors, 1)

    @property
    def matching_boxes_shape(self):
        w, h, _ = self.image_shape
        return (w / 32, h / 32, self.n_anchors, self.n_anchors)

    def _build_model(self):
        """Create YOLO v2 model CNN body.

        Argument
            inputs: input tensor
            n_anchors: number of bouding box anchors
            n_classes: number of classes
        """
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[
                                     None] + list(self.image_shape))
        conv13, darknet_output = darknet19(self.inputs)
        conv20 = conv2d_bn_leaky(darknet_output, 1024, 3)
        conv20 = conv2d_bn_leaky(conv20, 1024, 3)

        conv21 = conv2d_bn_leaky(conv13, 64, 1)
        conv21_reshaped = tf.space_to_depth(conv21, block_size=2)

        x = tf.concat([conv21_reshaped, conv20], axis=-1)
        x = conv2d_bn_leaky(x, 1024, 3)
        # Tha last FCN layer (= extracted image features for 1/32 scale)
        image_features = tf.layers.conv2d(
            x, self.n_anchors * (self.n_classes + 5), 1, padding='same')

        self.outputs = self._extract_bounding_boxes_layer(image_features)

    def _build_loss(self):
        self.gt_boxes = tf.placeholder(tf.float32, shape=(None, None, 5))
        self.gt_detectors_mask = tf.placeholder(
            tf.float32, shape=[None] + list(self.detectors_mask_shape))
        self.gt_matching_boxes = tf.placeholder(
            tf.float32, shape=[None] + list(self.matching_boxes_shape))
        loss = _yolo_loss_function(list(
            self.outputs) + [self.gt_boxes, self.gt_detectors_mask, self.gt_matching_boxes], self.anchors, self.n_classes)
        self.loss = loss

    def _build_train_op(self):
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    def _init_saver(self):
        self.saver = tf.train.Saver()

    def save(self, sess, checkpoint_dir, name):
        print('Saving model...')
        self.saver.save(sess, os.path.join(checkpoint_dir, name), self.global_step_tensor)
        print('saved.')

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def _extract_bounding_boxes_layer(self, image_features):
        """Convert final layer image features to bounding box parameters.

        Args:
            feats: Final convolutional layer features.
            anchors: Anchor box widths and heights.
            n_classes: Number of target classes.

        Returns:
            image_features (tf.Tensor): Raw image features
            box_xy (tf.Tensor): x, y box predictions adjusted by spatial
                                location in conv layer.
            box_wh (tf.Tensor): w, h box predictions adjusted by anchors and
                                conv spatial resolution.
            box_confidence (tf.Tensor): Probability estimate for whether each
                                        box contains any object.
            box_class_probs (tf.Tensor): Probability estimate for each box over
                                         class labels.
        """
        anchors_tensor = tf.reshape(tf.Variable(self.anchors), [
                                    1, 1, 1, self.n_anchors, 2])

        conv_dims = image_features.shape[1:3]
        conv_height_index = tf.range(0, conv_dims[0])
        conv_width_index = tf.range(0, conv_dims[1])
        conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])

        conv_width_index = tf.tile(tf.expand_dims(
            conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = tf.reshape(tf.transpose(conv_width_index), [-1])
        conv_index = tf.transpose(
            tf.stack([conv_height_index, conv_width_index]))
        conv_index = tf.reshape(
            conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = tf.cast(conv_index, image_features.dtype)

        def reshape_lambda(x):
            return tf.reshape(
                x, shape=[-1, conv_dims[0], conv_dims[1],
                          self.n_anchors, self.n_classes + 5])

        image_features_reshaped = tf.reshape(image_features,
                                             shape=[-1, conv_dims[0], conv_dims[1], self.n_anchors, self.n_classes + 5])
        conv_dims_reshaped = tf.cast(tf.reshape(
            conv_dims, [1, 1, 1, 1, 2]), image_features_reshaped.dtype)

        box_confidence = tf.sigmoid(
            image_features_reshaped[..., 4:5], name='box_confidence')
        box_class_probs = tf.nn.softmax(
            image_features_reshaped[..., 5:], name='box_class_probs')
        box_xy = tf.identity(
            (tf.sigmoid(
                image_features_reshaped[..., :2]) + conv_index) / conv_dims_reshaped,
            name='box_xy',
        )
        box_wh = tf.identity(tf.exp(
            image_features_reshaped[..., 2:4]) * anchors_tensor / conv_dims_reshaped, name='box_wh')

        return image_features_reshaped, box_xy, box_wh, box_confidence, box_class_probs

    def summary(self):
        print("=====> Prediction Model")
        self.model.summary()

        print()
        print("=====> Loss Model")
        self.model_loss.summary()

    def preprocess_true_boxes(self, true_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find detector in YOLO where ground truth box should appear.

        Args:
            true_boxes: List of ground truth boxes in form of `x`, `y`, `w`,
                        `h`, `class id`. Relative coordinates are in the range
                        [0, 1] indicating a percentage of the original image
                        dimensions.

        Returns:
            detectors_mask (np.ndarray):
                0/1 mask for detectors in form of `[conv_height, conv_width, n_anchors, 1]`
                that should be compared with a matching ground truth box.
            matching_true_boxes (np.ndarray):
                shape: `[conv_height, conv_width, n_anchors, n_box_params]` (`n_box_params` must be 5 here).
                Same shape as detectors_mask with the corresponding ground truth
                box adjusted for comparison with predicted parameters.
        """
        height, width, _ = self.image_shape
        n_anchors = self.n_anchors
        conv_height, conv_width = self.conv_height, self.conv_width
        # FIXME(agatan): Must be 5? (x, y, w, h, class)
        n_box_params = true_boxes.shape[1]
        detectors_mask = np.zeros(
            (conv_height, conv_width, n_anchors, 1), dtype=np.float32)
        matching_true_boxes = np.zeros(
            (conv_height, conv_width, n_anchors, n_box_params), dtype=np.float32)

        for box in true_boxes:  # type: np.ndarray
            # box is [x, y, w, h, class].
            assert box.shape == (5,)
            box_class = box[4]
            # box_geo is [conv_x, conv_y, conv_w, conv_h].
            box_geo = box[0:4] * \
                np.array([conv_width, conv_height, conv_width, conv_height])
            i = np.floor(box_geo[1]).astype('int')
            j = np.floor(box_geo[0]).astype('int')
            best_iou = 0
            best_anchor = 0
            # TODO(agatan): use np.argmax and np.apply
            for k, anchor in enumerate(self.anchors):
                # anchor is [w, h].
                box_maxes = box_geo[2:4] / 2.
                box_mins = -box_geo[2:4] / 2.
                anchor_maxes = anchor / 2.
                anchor_mins = -anchor / 2.

                intersect_maxes = np.maximum(box_maxes, anchor_maxes)
                intersect_mins = np.maximum(box_mins, anchor_mins)
                intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[0] * intersect_wh[1]
                box_area = box_geo[2] * box_geo[3]  # box's w * h
                anchor_area = anchor[0] * anchor[1]  # anchor's w * h
                iou = intersect_area / \
                    (box_area + anchor_area - intersect_area)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = k

            if best_iou > 0:
                detectors_mask[i, j, best_anchor] = 1
                # Create adjuested box.
                adjusted_box = np.array(
                    [box_geo[0] - j, box_geo[1] - i,
                     np.log(box_geo[2] / self.anchors[best_anchor][0]),
                     np.log(box_geo[3] / self.anchors[best_anchor][1]),
                     box_class
                     ], dtype=np.float32)
                matching_true_boxes[i, j, best_anchor] = adjusted_box

        return detectors_mask, matching_true_boxes

    def train(self, sess: tf.Session, image_data: np.ndarray, gt_boxes: np.ndarray, gt_detectors_mask: np.ndarray, gt_matching_boxes: np.ndarray):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.inputs: image_data,
            self.gt_boxes: gt_boxes,
            self.gt_detectors_mask: gt_detectors_mask,
            self.gt_matching_boxes: gt_matching_boxes,
        })



def _yolo_loss_function(args, anchors, n_classes):
    image_features, pred_xy, pred_wh, pred_confidence, pred_class_prob, \
        true_boxes, detectors_mask, matching_true_boxes = args

    pred_boxes = tf.concat(
        (tf.sigmoid(image_features[..., 0:2]), image_features[..., 2:4]),
        axis=-1)

    pred_xy = tf.expand_dims(pred_xy, 4)
    pred_wh = tf.expand_dims(pred_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = tf.shape(true_boxes)

    true_boxes = tf.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    best_ious = tf.reduce_max(iou_scores, axis=4)
    best_ious = tf.expand_dims(best_ious, axis=-1)

    object_detections = tf.cast(best_ious > 0.6, best_ious.dtype)

    no_object_scale = 1
    no_object_weights = no_object_scale * \
        (1 - object_detections) * (1 - detectors_mask)
    no_objects_loss = no_object_weights * tf.square(pred_confidence)
    objects_loss = 5 * detectors_mask * tf.square(1 - pred_confidence)
    confidence_loss = objects_loss + no_objects_loss

    matching_classes = tf.one_hot(
        tf.cast(matching_true_boxes[..., 4], tf.int32), n_classes)
    classification_loss = 1 * detectors_mask * \
        tf.square(matching_classes - pred_class_prob)
    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = 1 * detectors_mask * \
        tf.square(matching_boxes - pred_boxes)

    confidence_loss_sum = tf.reduce_sum(confidence_loss)
    tf.summary.scalar('confidence_loss', confidence_loss_sum)
    classification_loss_sum = tf.reduce_sum(classification_loss)
    tf.summary.scalar('classification_loss', classification_loss_sum)
    coordinates_loss_sum = tf.reduce_sum(coordinates_loss)
    tf.summary.scalar('coordinates_loss', coordinates_loss_sum)
    total_loss = 0.5 * (confidence_loss_sum +
                        classification_loss_sum + coordinates_loss_sum)

    tf.Print(total_loss,
             [total_loss, confidence_loss_sum, classification_loss_sum,
              coordinates_loss_sum],
             message='yolo_loss, confidence_loss, class_loss, box_coord_loss:')

    return total_loss


def preprocess_true_boxes(true_boxes, anchors, image_size):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    height, width = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros(
        (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (conv_height, conv_width, num_anchors, num_box_params),
        dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        box = box[0:4] * np.array(
            [conv_width, conv_height, conv_width, conv_height])
        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')
        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i,
                    np.log(box[2] / anchors[best_anchor][0]),
                    np.log(box[3] / anchors[best_anchor][1]), box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes


def print_summary():
    """Print network summary of YOLO v2
    """
    model = YOLO(
        image_shape=(416, 416, 3),
        classes=[str(i) for i in range(20)],
    )
    print(model.outputs)
    print(model.loss)
    # print(model.train_op)
    # model.summary()


if __name__ == '__main__':
    print_summary()
