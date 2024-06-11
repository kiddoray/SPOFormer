from .build import build_model_from_cfg
import sys
sys.path.append('../pointnet2_ops_lib')
sys.path.append('..')
import models.TopNet
import models.PoinTr
import models.GRNet
import models.PCN
import models.FoldingNet
import models.Snowflake
import models.PMPnet
import models.ASFM
import models.VRCNet
import models.ecg
import models.cascade
import models.ecg_shapenet55
import models.seedformer
import models.SPO
import models.SPO_seed
import models.SPOFormer
import models.proxyformer
import models.AnchorFormer