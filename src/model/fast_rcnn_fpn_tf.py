import tensorflow as tf
from keras import layers, Model

# ----- FPN backbone -----
def resnet_fpn_backbone(input_shape=(None, None, 3)):
    """
    Builds a ResNet50 backbone + FPN feature pyramid.
    Returns a dictionary of feature maps at multiple scales.
    """
    base_model = tf.keras.applications.ResNet50(
        include_top=False, input_shape=input_shape, weights='imagenet'
    )
    
    # Extract feature maps from specific layers
    layer_names = [
        'conv2_block3_out',  # C2
        'conv3_block4_out',  # C3
        'conv4_block6_out',  # C4
        'conv5_block3_out',  # C5
    ]
    layers_outputs = [base_model.get_layer(name).output for name in layer_names]
    c2, c3, c4, c5 = layers_outputs

    # Build lateral 1x1 convs for FPN
    p5 = layers.Conv2D(256, 1, name='fpn_p5')(c5)
    p4 = layers.Conv2D(256, 1, name='fpn_p4')(c4) + tf.image.resize(p5, tf.shape(c4)[1:3])
    p3 = layers.Conv2D(256, 1, name='fpn_p3')(c3) + tf.image.resize(p4, tf.shape(c3)[1:3])
    p2 = layers.Conv2D(256, 1, name='fpn_p2')(c2) + tf.image.resize(p3, tf.shape(c2)[1:3])
    
    # Smooth convs
    p2 = layers.Conv2D(256, 3, padding='same')(p2)
    p3 = layers.Conv2D(256, 3, padding='same')(p3)
    p4 = layers.Conv2D(256, 3, padding='same')(p4)
    p5 = layers.Conv2D(256, 3, padding='same')(p5)
    
    model = Model(inputs=base_model.input, outputs={'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5})
    return model


# ----- ROI Pooling (approximation) -----
def roi_align_single_feature_map(feature_map, rois, roi_size=(7, 7)):
    """
    Performs ROI pooling using crop_and_resize.
    rois: tensor of shape [N, 4] in (y1, x1, y2, x2) normalized coordinates
    """
    # The 3rd argument (box_indices) assumes all ROIs come from the same image (index 0)
    box_indices = tf.zeros((tf.shape(rois)[0],), dtype=tf.int32)
    return tf.image.crop_and_resize(
        feature_map, boxes=rois, box_indices=box_indices,
        crop_size=roi_size
    )


# ----- Classification Head -----
def build_classification_head(input_dim, num_classes=2):
    return tf.keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)  # logits
    ])


# ----- FasterRCNN_FPN TensorFlow version -----
class FasterRCNN_FPN_TF(tf.keras.Model):
    """
    TensorFlow version of the FasterRCNN_FPN parking lot classifier.
    """
    def __init__(self, roi_res=7, pooling_type='square', num_classes=2, input_shape=(None, None, 3)):
        super().__init__()
        self.roi_res = roi_res
        self.pooling_type = pooling_type
        
        # Backbone (ResNet-FPN)
        self.backbone = resnet_fpn_backbone(input_shape)
        
        # Classification head
        in_channels = 256 * (roi_res ** 2)
        self.class_head = build_classification_head(in_channels, num_classes)

    def call(self, image, rois, training=False):
        # Extract features
        features = self.backbone(image, training=training)
        
        # Use the last feature pyramid map for simplicity (you could combine them)
        feature_map = features['p4']  # [B, H, W, C]
        
        # Perform ROI pooling
        pooled_rois = roi_align_single_feature_map(feature_map, rois, (self.roi_res, self.roi_res))
        
        # Flatten and classify
        pooled_flat = tf.reshape(pooled_rois, [tf.shape(pooled_rois)[0], -1])
        class_logits = self.class_head(pooled_flat)
        return class_logits