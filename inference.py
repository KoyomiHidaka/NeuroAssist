import cv2
from ultralytics import YOLO
import pyautogui
import numpy as np
import torch
from mss import mss

# Загрузка дообученной модели YOLOv8
model = YOLO('runs/detect/train/weights/best.pt')
# Настройка захвата экрана
sct = mss()

def process_frame(frame):
    # Преобразование кадра в нужный формат для модели YOLOv8
    results = model(frame)
    return results

def main():
    while True:
        # Захват экрана
        screen = np.array(sct.grab(sct.monitors[0]))
        screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

        # Обработка кадра моделью
        results = process_frame(screen_rgb)

        # Получение результатов распознавания
        for detection in results[0].boxes:  # Использование 'boxes' для доступа к коробкам
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Извлечение координат
            conf = detection.conf[0]
            if conf > 0.5:  # Порог уверенности
                # Координаты центра кнопки
                #center_x = int((x1 + x2) / 2)
                #center_y = int((y1 + y2) / 2)

                # Выполнение клика мышкой
                pyautogui.click()
                #pyautogui.press('space')

        # Выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
