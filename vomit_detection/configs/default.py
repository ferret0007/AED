from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix =''
_CN.batch_size = 2
_CN.sum_freq = 100
_CN.val_freq = 10000
_CN.data_balance = True
_CN.cache_data = False
_CN.critical_params = ["batch_size", "frame_num", "data_balance"]
_CN.mixed_precision = False

_CN.restore_ckpt = "r3d18_bs24_frame30_iter3k_dataall/base/batch_size[24]frame_num[30]data_balance[True](01_14_22_31)/final.pth"
_CN.model = 'base'

_CN.frame_num = 30

##########################################
# base network
_CN.base = CN()
_CN.base.critical_params = []

# swin 3D
_CN.swin3d = CN()
_CN.swin3d.critical_params = []

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'

_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 25e-5
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 10000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
