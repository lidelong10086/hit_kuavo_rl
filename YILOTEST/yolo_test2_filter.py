import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# 初始化卡尔曼滤波器
def init_kalman():
    # dim_x=4: 状态量有4个 (x, y, dx, dy) 分别是中心点坐标和运动速度
    # dim_z=2: 观测值有2个 (x, y) 也就是YOLO给出的坐标
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    # 状态转移矩阵 (假设是匀速运动模型)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # 观测矩阵 (我们只能观测到位置 x, y)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    
    # 测量噪声矩阵 (信任YOLO的程度，数值越小越信任)
    kf.R *= 10
    
    # 过程激励噪声 (允许模型运动状态发生变化的程度)
    kf.P *= 1000
    kf.Q[-1,-1] *= 0.01
    kf.Q[-2,-2] *= 0.01
    
    return kf

# 加载模型
model = YOLO('/home/xie/HITIRC_KUAVO_RL/runs/detect/door_project/my_handler_model2/weights/best.pt')
cap = cv2.VideoCapture(0)

kf = init_kalman()
initialized = False # 标记是否已经锁定了第一个目标

while True:
    ret, frame = cap.read()
    if not ret: break

    # YOLO检测
    results = model(frame, conf=0.5) # 设置置信度阈值
    boxes = results[0].boxes
    
    # 卡尔曼预测阶段
    kf.predict()
    
    current_center = None
    
    # 如果YOLO检测到了物体
    if len(boxes) > 0:
        # 只取置信度最高的第一个框
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if not initialized:
            # 第一次检测到物体，初始化坐标
            kf.x = np.array([[cx], [cy], [0], [0]])
            initialized = True
        
        # 卡尔曼更新阶段 (用YOLO的结果修正预测)
        kf.update(np.array([[cx], [cy]]))
        current_center = (int(cx), int(cy))
        
        # 画YOLO的原始蓝色框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "YOLO Detection", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 显示卡尔曼滤波的结果
    # kf.x[0]和kf.x[1]就是卡尔曼推算出的X和Y
    kf_x, kf_y = int(kf.x[0]), int(kf.x[1])
    
    # 画卡尔曼滤波预测的绿色圆点（即使YOLO丢失，这个圆点也会动）
    if initialized:
        cv2.circle(frame, (kf_x, kf_y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"Tracked: {kf_x}, {kf_y}", (kf_x + 10, kf_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if len(boxes) == 0:
            cv2.putText(frame, "YOLO LOST! Predicting...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Kalman Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()