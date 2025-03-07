import math
import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor

import ltr.models.target_classifier.transformer as transformer
import ltr.models.target_classifier.transformer_three as transformer_three
import ltr.models.target_classifier.transformer_add as transformer_add
import ltr.models.target_classifier.transformer_three_add as transformer_three_add
import ltr.models.target_classifier.transformer_three_add_v as transformer_three_add_v
import ltr.models.target_classifier.transformer_two_add as transformer_two_add

import ltr.models.target_classifier.transformer_no_adaptive as transformer_no_adaptive
from ltr.models.target_classifier.multihead_attention import MultiheadAttention


class DiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, classifier, bb_regressor, classification_layer, bb_regressor_layer,
                 qxm):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))
        self.drop_path_prob = 0.0
        self.qxm = qxm


    def forward(self, train_imgs, test_imgs, train_bb, train_label, test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""
        
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        num_img_train = train_imgs.shape[0]
        num_img_test = test_imgs.shape[0]

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        train_feat_clf = train_feat_clf.reshape(num_img_train, -1, *train_feat_clf.shape[-3:])
        test_feat_clf = test_feat_clf.reshape(num_img_test, -1, *test_feat_clf.shape[-3:])

        # self.plt_images_save(train_imgs)
  
        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_label, train_bb, qxm = self.qxm,
                                        *args, **kwargs)

        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        try:
            if layers[0]== 'layer2' and layers[1] == 'layer3' and self.feature_extractor.patch_embed is not None:
                layer_dim = 384
                return self.feature_extractor(im, layer_dim)
            else:
                return self.feature_extractor(im, layers)
        except:
            return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})



class DiMPnet_mvit(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, classifier, bb_regressor, classification_layer, bb_regressor_layer,channel_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer,
                                                                         str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))
        self.drop_path_prob = 0.0
        self.channel_layer = channel_layer


    def forward(self, train_imgs, test_imgs, train_bb, train_label, test_proposals, *args, **kwargs):
       
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        num_img_train = train_imgs.shape[0]
        num_img_test = test_imgs.shape[0]

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        train_feat_clf = train_feat_clf.reshape(num_img_train, -1, *train_feat_clf.shape[-3:])
        test_feat_clf = test_feat_clf.reshape(num_img_test, -1, *test_feat_clf.shape[-3:])

        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_label, train_bb, #qxm=self.qxm,
                                        *args, **kwargs)

        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None, channel_layer=None):
        if layers is None:
            layers = self.output_layers
        if channel_layer is None:
            channel_layer = self.channel_layer
        return self.feature_extractor(im, layers, channel_layer)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})




@model_constructor
def dimpnet_mvit(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=384, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(96, 96), iou_inter_dim=(96, 96),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=(),
              yaml_type=None, optimize_filter='SDGN',channel_layer=[192,384],
              pre_dim=1, conv2_dim=0.5, transformer_type='adaptive_three_add',
              cascade_level =2):
    pretrained_path = './Backbone_MViT/MViTv2_' + yaml_type + '_in1k.pth'
    yaml_path = './Backbone_MViT/configs/test/MVITv2_' + yaml_type + '_test.yaml'
    # Backbone
    backbone_net = backbones.mvit_transformer_backbone(yaml_path=yaml_path)
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    backbone_net.load_state_dict(checkpoint['model_state'])

    # layer0 to layer4
    # Feature normalization
    if out_feature_dim is None:
        out_feature_dim = channel_layer[-1]
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = int(channel_layer[-2] / 2)
        # feature_dim = 96#256
    elif classification_layer == 'layer4':
        feature_dim = int(channel_layer[-1] / 2)
        # feature_dim = 768#512
    else:
        raise Exception
    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim, num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale, out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm, feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    if optimize_filter == 'SDGN':
        optimizer = clf_optimizer.DiMPTemporalReSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride, init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins, bin_displacement=bin_displacement, mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act, detach_length=detach_length)
    else:
        return
    ### Transformer
    init_transformer = transformer.Transformer(d_model=channel_layer[-1], nhead=1, num_layers=1)#512
    
    # The classifier module #
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor, transformer=init_transformer)

    # Bounding box regressor
    bb_regressor = bbmodels.SwinIoUNet(input_dim=channel_layer, pred_input_dim=[feature_dim,feature_dim], pred_inter_dim=[feature_dim,feature_dim])
    net = DiMPnet_mvit(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                      classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'],
                      channel_layer=channel_layer)
    return net
