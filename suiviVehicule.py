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

# Dictionnaire pour suivre les véhicules avec une mémoire étendue
vehicle_ids = {}
next_vehicle_id = 1
memory_frames = 3  # Nombre de frames de mémoire pour le suivi
vehicle_memory = {}
frame_last_seen = {}  # Stocker la dernière frame vue pour chaque véhicule

# Obtenir le FPS de la vidéo pour ajuster l'attente entre les frames
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / fps  # Temps entre chaque frame en secondes
prev_time = time.time()

# Définir les zones de confidentialité (deux triangles)
def is_in_confidential_zone(point, triangles):
    def is_inside_triangle(pt, tri):
        x, y = pt
        x1, y1, x2, y2, x3, y3 = tri
        # Calculer les aires des sous-triangles
        area = abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)
        area1 = abs((x*(y2-y3) + x2*(y3-y) + x3*(y-y2)) / 2.0)
        area2 = abs((x1*(y-y3) + x*(y3-y1) + x3*(y1-y)) / 2.0)
        area3 = abs((x1*(y2-y) + x2*(y-y1) + x*(y1-y2)) / 2.0)
        return area == (area1 + area2 + area3)

    for tri in triangles:
        if is_inside_triangle(point, tri):
            return True
    return False

# Coordonnées des triangles (x, y pour chaque sommet)
confidential_zones = [
    (100, 0, 550, 600, 0, 600),  # Triangle 1
    (400, 0, 1200, 0, 1200, 600)  # Triangle 2
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    height, width, _ = frame.shape

    frame_modulo = 7

    # Traiter une image sur 7 (4 fps)
    if frame_count % frame_modulo == 0:
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

                    # Vérifier si le centre de la boîte est dans une zone de confidentialité
                    if is_in_confidential_zone((center_x, center_y), confidential_zones):
                        continue

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Assigner un ID à chaque détection
        current_ids = {}
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                center_x = x + w // 2
                center_y = y + h // 2

                # Vérifier si la détection correspond à un véhicule existant
                found_id = None
                for vid, pos in vehicle_ids.items():
                    prev_center_x, prev_center_y = pos
                    if abs(center_x - prev_center_x) < 50 and abs(center_y - prev_center_y) < 50:
                        found_id = vid
                        break

                # Si aucun ID existant ne correspond, vérifier dans la mémoire
                if found_id is None:
                    for vid, positions in vehicle_memory.items():
                        if any(abs(center_x - px) < 50 and abs(center_y - py) < 50 for px, py in positions):
                            found_id = vid
                            break

                # Si toujours aucun ID, en créer un nouveau
                if found_id is None:
                    found_id = next_vehicle_id
                    next_vehicle_id += 1

                current_ids[found_id] = (center_x, center_y)
                frame_last_seen[found_id] = frame_count

        # Mettre à jour les IDs des véhicules
        vehicle_ids = current_ids

        # Mettre à jour la mémoire des positions
        for vid, pos in current_ids.items():
            if vid not in vehicle_memory:
                vehicle_memory[vid] = []
            vehicle_memory[vid].append(pos)
            if len(vehicle_memory[vid]) > memory_frames:
                vehicle_memory[vid].pop(0)

        # Supprimer les véhicules qui ont disparu depuis plus de 5 frame analysé (soit 5 * frame modulo, 35 frames)
        vehicles_to_remove = [vid for vid, last_seen in frame_last_seen.items() if frame_count - last_seen > 5*frame_modulo]
        for vid in vehicles_to_remove:
            if vid in vehicle_ids:
                del vehicle_ids[vid]
            if vid in vehicle_memory:
                del vehicle_memory[vid]
            if vid in frame_last_seen:
                del frame_last_seen[vid]

    # Affichage des positions, boîtes et IDs
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            center_x = x + w // 2
            center_y = y + h // 2

            # Dessiner les boîtes et afficher les positions et IDs
            found_id = None
            for vid, pos in vehicle_ids.items():
                if pos == (center_x, center_y):
                    found_id = vid
                    break

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"ID: {found_id}, X: {center_x}, Y: {center_y}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dessiner les zones de confidentialité
    for tri in confidential_zones:
        pts = np.array([[tri[0], tri[1]], [tri[2], tri[3]], [tri[4], tri[5]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Calculer le temps réel pour ajuster le délai
    current_time = time.time()
    elapsed_time = current_time - prev_time
    delay = max(int((frame_time - elapsed_time) * 1000), 1)  # Calcul du délai en ms
    prev_time = current_time

    # Afficher la vidéo avec annotations
    cv2.imshow("Frame", frame)
    if cv2.waitKey(delay) != -1:
        break

cap.release()
cv2.destroyAllWindows()
