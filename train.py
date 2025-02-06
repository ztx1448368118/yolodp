import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolodp.yaml')
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
                # resume=True, # 
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
