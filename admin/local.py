class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/models'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'  # Directory for tensorboard files.
        self.lasot_dir = '/data/trackingdata/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/data/trackingdata/GOT10K/train'
        self.trackingnet_dir = ''
        self.coco_dir = '/data/trackingdata/coco'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.SV248s_dir = '/data/zxin/trackingdata/SV248_2023'
        self.skysat_dir = '/data/zxin/trackingdata/SkySat'
        self.viso_dir = '/data/zxin/trackingdata/VISOdataset'
        self.trackingnet_dir = '/data/trackingdata/TrackingNet'

        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''