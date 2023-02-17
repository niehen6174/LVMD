import time
import os
import torch
from options.test_options import TestOptions
from models.models import create_model
import struct

opt = TestOptions().parse()

model = create_model(opt)

if os.path.isfile('DeblurGan.wts'):
    print('Already, deblurgan.wts file exists.')
    batch_size = 1  # 批处理大小
    input_shape = (3, 720, 720)  # 输入数据,改成自己的输入shape

    # #set the model to inference mode

    model = model.netG
    model.eval()
    x = torch.randn(batch_size, *input_shape)  # 生成张量
    export_onnx_file = "test.onnx"  # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      opset_version=13,
                      do_constant_folding=False,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"],  # 输出名
                      dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                    "output": {0: "batch_size"}})
else:
    print('making DeblurGan.wts file ...')
    f = open("DeblurGan.wts", 'w')
    f.write("{}\n".format(len(model.netG.model.state_dict().keys())))
    for k, v in model.netG.model.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
