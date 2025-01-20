import cv2
import numpy as np
import time

# Charger YOLOv4-tiny
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Lecture de la vidéo
cap = cv2.VideoCapture("Video_route_1.mp4")
frame_count = 0
boxes = []
confidences = []
class_ids = []

# Obtenir le FPS de la vidéo pour ajuster l'attente entre les frames
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / fps  # Temps entre chaque frame en secondes
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    height, width, _ = frame.shape

    # Traiter une image sur 15 (2 fps)
    if frame_count % 15 == 0:
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Analyse des détections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 2:  # Classe '2' pour voiture
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Affichage des positions et boîtes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            center_x = x + w // 2
            center_y = y + h // 2

            # Dessiner les boîtes et afficher les positions
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"X: {center_x}, Y: {center_y}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculer le temps réel pour ajuster le délai
    current_time = time.time()
    elapsed_time = current_time - prev_time
    delay = max(int((frame_time - elapsed_time) * 1000), 1)  # Calcul du délai en ms
    prev_time = current_time

    # Afficher la vidéo avec annotations
    cv2.imshow("Frame", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
