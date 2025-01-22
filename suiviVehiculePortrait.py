import cv2
import numpy as np
import time
# Lecture de la vidéo
import yt_dlp

# Définir la classe Vehicle
class Vehicle:
    def __init__(self, vehicle_id, position):
        self.id = vehicle_id
        self.position = position  # (x, y)
        self.next_position = (0, 0)  # (x, y)
        self.speed = (0, 0)  # (vx, vy)
        self.angle = 0.0  # Direction en radians
        self.last_seen = 0  # Dernière frame où le véhicule a été vu
        self.angle_buffer = []  # Buffer des derniers angles
        self.direction = "NONE" # Direction sur la route
        self.plaque = "NULL" # Plaque d'immatriculation

    def update_position(self, position):
        self.position = position

    def update_next_position(self, next_position):
        self.next_position = next_position

    def update_speed(self, speed):
        self.speed = speed

    def update_angle(self, angle):
        self.angle_buffer.append(angle)
        if len(self.angle_buffer) > 5:  # Supposons un buffer de 5 valeurs
            self.angle_buffer.pop(0)
        self.angle = sum(self.angle_buffer) / len(self.angle_buffer)  # Moyenne des angles

# Charger YOLOv4-tiny
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

"""
def get_youtube_stream_url(youtube_url, format_id='95'):  # 1280x720, format ID 95
    ydl_opts = {
        'format': format_id,  # Choisir un format spécifique
        'quiet': False,       # Activer les logs pour le débogage
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            formats = info_dict.get('formats', [])
            for f in formats:
                if f['format_id'] == format_id:
                    return f['url']
            raise Exception(f"Format ID {format_id} non trouvé.")
    except Exception as e:
        print(f"Erreur lors de l'extraction du flux : {e}")
        raise Exception("Impossible de récupérer l'URL du flux YouTube.")
"""

ii = 0

def detect_license_plate(image):
    """
    Fausse fonction pour détecter une plaque d'immatriculation.

    Args:
        image (numpy.ndarray): Image d'entrée (par exemple, une frame d'une vidéo).

    Returns:
        str: Une plaque d'immatriculation fictive.
    """
    # Ici, on simule simplement une détection
    if(ii % 10 == 0):
        plaque = "NULL"
    else:
        plaque = "AB" + ii + "3CD"
    return plaque

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

"""
youtube_url = "https://www.youtube.com/watch?v=EPKWu223XEg"  # Remplacez par l'ID du flux
try:
    stream_url = get_youtube_stream_url(youtube_url)
    cap = cv2.VideoCapture(stream_url)
except Exception as e:
    print(f"Erreur : {e}")
    exit()
"""
cap = cv2.VideoCapture("rush/Video1.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution originale : {width}x{height}")


frame_count = 0

# Dictionnaire pour stocker les véhicules
vehicles = {}
next_vehicle_id = 1

# Obtenir le FPS de la vidéo pour ajuster l'attente entre les frames
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / fps  # Temps entre chaque frame en secondes
prev_time = time.time()

# Coordonnées des triangles (x, y pour chaque sommet)
confidential_zones = [
    #(60, 0, 650, 720, 0, 720),  # Triangle 1 yx yx yx
    #(120, 0, 1280, 0, 1050, 720)  # Triangle 2 yx yx yx
]

boxes = []
class_ids = []
confidences = []

paused = False

while cap.isOpened():
    if not paused:

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
            
            # Réinitialisation uniquement pour les frames analysées
            boxes.clear()
            confidences.clear()
            class_ids.clear()
                

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
                    for vid, vehicle in vehicles.items():
                        prev_center_x, prev_center_y = vehicle.next_position
                        prev_speed_x, prev_speed_y = vehicle.speed
                        if abs(center_x - prev_center_x) < 50 and abs(center_y - prev_center_y) < 50:
                            found_id = vid
                            break

                    # Si aucun ID existant ne correspond, en créer un nouveau
                    if found_id is None:
                        found_id = next_vehicle_id
                        next_vehicle_id += 1
                        vehicles[found_id] = Vehicle(found_id, (center_x, center_y))

                    # Mettre à jour le véhicule
                    vehicle = vehicles[found_id]

                    if frame_count - vehicle.last_seen > 0:
                        prev_x, prev_y = vehicle.position
                        speed_x = (center_x - prev_x) / (frame_count - vehicle.last_seen)
                        speed_y = (center_y - prev_y) / (frame_count - vehicle.last_seen)
                    else:
                        speed_x = 0
                        speed_y = 0

                    #Calcule de l'estimation de la prochaine position
                    next_x = center_x + speed_x * frame_modulo
                    next_y = center_y + speed_y * frame_modulo
                    vehicle.update_next_position((next_x, next_y))

                    vehicle.update_position((center_x, center_y))
                    vehicle.update_speed((speed_x, speed_y))

                    if speed_x != 0 or speed_y != 0:
                        angle = np.arctan2(speed_y, speed_x)
                        vehicle.update_angle(angle)
                        #On recupére l'angle car il a été moyenné
                        angle = vehicle.angle
                        if(angle > -0.80) & (angle < 2.35):
                            direction = "UP"
                        else:
                            direction = "DOWN"
                        vehicle.direction = direction
                    vehicle.last_seen = frame_count

                    #On découpe la voiture dans l'image pour l'envoyer dans la fonction de reconnaissance de plaque

                    voitureDecoupe = frame[y:y+h, x:x+w]
                    #cv2.imshow("voiture", voitureDecoupe)
                    #cv2.waitKey(0)
                    plaque = detect_license_plate(voitureDecoupe)
                    if(plaque != "NULL"):
                        print("Plaque détectée : ", plaque)
                        vehicle.plaque = plaque
                
            # Supprimer les véhicules qui ont disparu depuis plus de  5 frame analysé (soit 5 * frame_modulo, 35 réel frames)
            vehicles_to_remove = [vid for vid, vehicle in vehicles.items() if frame_count - vehicle.last_seen > 5 * frame_modulo]
            for vid in vehicles_to_remove:
                del vehicles[vid]
            
        # Affichage des positions, boîtes, IDs et vitesses
        if len(boxes) > 0:  # Vérifie que la liste n'est pas vide
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Dessiner les boîtes et afficher les positions, IDs et vitesses
                    found_id = None
                    for vid, vehicle in vehicles.items():
                        if vehicle.position == (center_x, center_y):
                            found_id = vid
                            break

                    vehicle = vehicles.get(found_id)
                    if vehicle:
                        speed_x, speed_y = vehicle.speed
                        angle = vehicle.angle
                        if(vehicle.direction == "NONE"):
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        elif(vehicle.direction == "UP"):
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        if(vehicle.plaque != "NULL"):
                            label = f"ID: {vehicle.plaque}, X: {center_x}, Y: {center_y}, Vx: {speed_x:.2f}, Vy: {speed_y:.2f}, Angle: {angle:.2f}"
                        else:
                            label = f"ID: {vehicle.id}, X: {center_x}, Y: {center_y}, Vx: {speed_x:.2f}, Vy: {speed_y:.2f}, Angle: {angle:.2f}"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        #Dessiner la direction du véhicule
                        x2 = int(center_x + 50 * np.cos(angle))
                        y2 = int(center_y + 50 * np.sin(angle))
                        cv2.line(frame, (center_x, center_y), (x2, y2), (0, 0, 255), 2)
                        
                        #Dessiner la vitesse
                        x2 = int(center_x + speed_x*frame_modulo)
                        y2 = int(center_y + speed_y*frame_modulo)
                        cv2.arrowedLine(frame, (center_x, center_y), (x2, y2), (255, 0, 0), 2)

        cv2.putText(frame, f"Boxes: {len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

        # Réduire la taille de l'image par un facteur de 2
        scale_factor = 0.3  # Divise par 2
        frame_resized = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        frame_rotated = cv2.rotate(frame_resized, cv2.ROTATE_90_CLOCKWISE)
        # Afficher l'image redimensionnée
        cv2.imshow("Frame", frame_rotated)


    key = cv2.waitKey(delay) & 0xFF
    if key == 27:  # Échapper pour quitter
        break
    elif key == 32:  # Barre d'espace pour mettre en pause/reprendre
        paused = not paused

cap.release()
cv2.destroyAllWindows()
