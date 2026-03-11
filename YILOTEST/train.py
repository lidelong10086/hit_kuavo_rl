from ultralytics import YOLO



#加载模型

model = YOLO('yolov8n.pt')



#开始训练

results = model.train(

data='door.yaml',

epochs=50,

imgsz=640,

device='cpu',

workers=0,

batch=8, 

project='door_project',

name='my_handler_model'

)