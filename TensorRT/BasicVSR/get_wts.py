import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from mmcv import Config
from mmcv.runner import  load_checkpoint
from mmedit.models import build_model
import struct
# https://gitee.com/zfr9b/mmediting

def main():
    #https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth 
    checkpoint_file = "./basicvsr_reds4_20120409-0e599677.pth"
    #https://gitee.com/zfr9b/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_reds4.py
    config_file = "basicvsr_reds4.py"

    cfg = Config.fromfile(config_file)
    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, checkpoint_file, map_location='cpu')
    f = open("./BasicVSR_layerName.wts", 'w')
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        #f.write("layer name: " + k + ", shape:" + str(v.shape))
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    

   
if __name__ == '__main__':
    main()
