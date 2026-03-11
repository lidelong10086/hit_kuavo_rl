from ultralytics import YOLO
import cv2

# 加载模型（可换模型）
model = YOLO("yolov8n.pt")
# model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取摄像头失败/摄像头已断开")
        break

    #检测
    results = model(frame)

    #提取检测框并计算中心坐标
    boxes = results[0].boxes  # 获取所有检测框的Boxes对象

    if boxes is not None and len(boxes) > 0:  # 先判断是否检测到物体
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0] # 四角坐标
            center_x = int((x1 + x2) / 2)  # 中心x坐标
            center_y = int((y1 + y2) / 2)  # 中心y坐标
            x_center, y_center, width, height = box.xywh[0]

            # 类别和置信度
            cls_id = int(box.cls[0])
            class_name = results[0].names[cls_id]
            confidence = float(box.conf[0])

            # 输出
            print(f"检测到: {class_name} "
                  f"置信度: {confidence:.2f} "
                  f"边界框: ({x1}, {y1}), ({x2}, {y2}) "
                  f"中心坐标: ({x_center}, {y_center})")

            # 在画面上画出中心坐标点
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # 红色实心点
            # 在中心坐标整数显示
            cv2.putText(frame, f"({center_x},{center_y})", (center_x+10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 画检测框
    annotated_frame = results[0].plot(img=frame)  # 基于带中心圆点的frame画框

    # 显示画面
    cv2.imshow("YOLO Detection (with Center Point)", annotated_frame)

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()