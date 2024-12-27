#!/usr/bin/python3
from ultralytics import YOLOv10
import cv2
import os

# 设置输入和保存目录
source = 'new_input3/person_horse_V3.mp4'
save_dir_p = './output_frames'
save_dir_v = './output_video'
os.makedirs(save_dir_p, exist_ok=True)
os.makedirs(save_dir_v, exist_ok=True)

# 加载 YOLOv10 模型
model = YOLOv10.from_pretrained('jameslahm/yolov10x')

# Open the input video
cap = cv2.VideoCapture(source)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for output video
out = cv2.VideoWriter(os.path.join(save_dir_v, 'output_video.mp4'), fourcc, fps, (frame_width, frame_height))

# 运行模型预测
results = model.predict(source=source, classes=[0, 17], imgsz=1280, conf=0.3, iou=0.7)

# 遍历所有预测结果
for i, result in enumerate(results):
    # 获取原始图像
    orig_image = result.orig_img.copy()  # 拷贝原始图像以进行修改
    
    # 获取检测到的类别
    if result.boxes is not None:
        detected_classes = result.boxes.cls.cpu().numpy()  # 将张量转换为 NumPy 数组
        class_human_count = (detected_classes == 0).sum()  # 计算“人”的数量
        class_horse_count = (detected_classes == 17).sum()  # 计算“马”的数量

        # 绘制边界框和标签
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), detected_classes, result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)  # 将边界框坐标转换为整数
            class_name = result.names[int(cls)]  # 获取类别名称

            if class_name == "person":
                color = (0, 0, 255)  # 红色
            elif class_name == "horse":
                color = (128, 0, 128)  # 紫色
            else:
                color = (255, 255, 255)  # 白色
            
            # 画出边界框
            cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, 4)
            
            # 绘制类别和置信度
            label = f"{class_name} {conf:.2f}"
            cv2.putText(orig_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color,3)

    else:
        class_human_count = 0
        class_horse_count = 0

    # 在帧上绘制学号和各个类的检测数量
    text = f"student ID: 312512049 | person: {class_human_count} | : {class_horse_count}"
    purple_color = (128, 0, 128)  # 紫色
    red_color = (0, 0, 255)  # 红色
    cv2.putText(orig_image, f"student ID: 312512049", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, purple_color, 2)
    cv2.putText(orig_image, f"person: {class_human_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2)
    cv2.putText(orig_image, f"horse: {class_horse_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, purple_color, 2)

    # 保存修改后的图像或帧
    frame_file_path = os.path.join(save_dir_p, f'frame_{i}.jpg')
    cv2.imwrite(frame_file_path, orig_image)
    out.write(orig_image)

cap.release()
out.release()
cv2.destroyAllWindows()
# 释放资源