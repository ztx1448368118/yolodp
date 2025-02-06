import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 指定显卡和多卡训练问题 统一都在<YOLOV8V10配置文件.md>下方常见错误和解决方案。
# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。

if __name__ == '__main__':
    model = YOLO('autodl-tmp/ultralytics-main1009/ultralytics/cfg/models/v3/yolov3-tiny.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='//root/autodl-tmp//ultralytics-main//dataset//data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=64,
                close_mosaic=10,
                workers=16,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )