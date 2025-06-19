import cv2
import easyocr
from collections import defaultdict
from ultralytics import YOLO
import re
import Levenshtein
import os


CONFIDENCE_THRESHOLD = 0.3
MIN_PLATE_LENGTH = 5
    
reader = easyocr.Reader(
    lang_list=['en'],
    recog_network='standard',
    model_storage_directory='./weights/EasyOCR',
    download_enabled=True
)

def correct_plate(plate):
    replace_map = {
        'S': '5',
        'Y': 'У',
        'N': 'Н',
        'R': 'Р'
    }
    corrected = [replace_map.get(c, c) for c in plate]
    return ''.join(corrected)

def normalize_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9 ]', '', text)
    text = text.replace('O', '0').replace('I', '1').replace('L', '1')
    return text.strip()

def cleaned_len(s):
    return len(re.sub(r'[^A-Z0-9]', '', s))

def cluster_similar_plates_graph(plates_dict, max_dist=2):
    plates = list(plates_dict.keys())
    n = len(plates)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if Levenshtein.distance(plates[i], plates[j]) <= max_dist]

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for i, j in edges:
        union(i, j)

    clusters_dict = defaultdict(list)
    for idx, plate in enumerate(plates):
        root = find(idx)
        clusters_dict[root].append((plate, plates_dict[plate]))
    return list(clusters_dict.values())

def format_timestamp(seconds_float):
    h = int(seconds_float // 3600)
    m = int((seconds_float % 3600) // 60)
    s = int(seconds_float % 60)
    ms = int((seconds_float - int(seconds_float)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def main(input_path, output_path, yolo_path):
    plate_confidences = {}
    
    model = YOLO(yolo_path)
    IS_IMAGE = input_path.lower().endswith(('.jpg', '.jpeg', '.png'))
    
    if IS_IMAGE:

        frame = cv2.imread(input_path)
        results = model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                ocr_results = reader.readtext(roi)
                
                best_text = ""
                best_conf = 0.0
                for ocr_res in ocr_results:
                    text, _, conf = ocr_res[1], ocr_res[0], ocr_res[2]
                    if conf > best_conf:
                        best_text = text
                        best_conf = conf
                norm_text = normalize_plate(best_text)
                if best_conf > CONFIDENCE_THRESHOLD and norm_text:
                    plate_confidences[norm_text] = {
                        "confidence": best_conf,
                        "timestamp": "image"
                    }

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, norm_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        cv2.imwrite(output_path, frame)
        return plate_confidences

    else:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = frame_count / fps
            timestamp_str = format_timestamp(timestamp_sec)

            results = model(frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    roi = frame[y1:y2, x1:x2]
                    ocr_results = reader.readtext(roi)

                    best_text = ""
                    best_ocr_conf = 0.0
                    for ocr_res in ocr_results:
                        text, _, ocr_conf = ocr_res[1], ocr_res[0], ocr_res[2]
                        if ocr_conf > best_ocr_conf:
                            best_text = text
                            best_ocr_conf = ocr_conf

                    norm_text = normalize_plate(best_text)
                    if best_ocr_conf > CONFIDENCE_THRESHOLD and norm_text:
                       if norm_text not in plate_confidences or plate_confidences[norm_text]['confidence'] < best_ocr_conf:
                           plate_confidences[norm_text] = {
                                "confidence": best_ocr_conf,
                                "timestamp": timestamp_str
                            }

                    label = f"{norm_text} ({best_ocr_conf:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

    clusters = cluster_similar_plates_graph(plate_confidences, max_dist=2)
    output_log_path = "./datatest/plates/plates_clusters_output.txt"

    with open(output_log_path, "w", encoding="utf-8") as f:

        cluster_number = 1
        found_any = False
        best_plates_overall = []

        for cluster in clusters:
            filtered = [(text, data) for text, data in cluster if len(text) >= MIN_PLATE_LENGTH]
            if not filtered:
                continue

            found_any = True
            f.write(f"Кластер #{cluster_number}\n")
            filtered.sort(key=lambda x: x[1]['confidence'], reverse=True)

            for i, (text, data) in enumerate(filtered, 1):
                f.write(f"{i}. {text} (доверие: {data['confidence']:.4f}, время: {data['timestamp']})\n")

            best_long_plate = None
            best_conf = -1
            for text, data in filtered:
                if cleaned_len(text) >= 8 and data['confidence'] > best_conf:
                    best_long_plate = (text, data)
                    best_conf = data['confidence']

            if not best_long_plate:
                best_long_plate = filtered[0]

            corrected_text = correct_plate(best_long_plate[0])
            corrected_plate = (corrected_text, best_long_plate[1])
            best_plates_overall.append(corrected_plate)

            cluster_number += 1

        if not found_any:
            f.write("Номера не найдены.\n")
    if best_plates_overall:
        plates_texts = {}
        for i, (text, data) in enumerate(best_plates_overall, 1):
            plates_texts[text] = {'confidence': round(float(data['confidence']), 4), 'timestamp': data['timestamp']}
        return plates_texts
    else:
        return 0
