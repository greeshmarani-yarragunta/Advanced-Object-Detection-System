import os
import time
import threading
import io
from queue import Queue, Empty
from collections import defaultdict

from flask import (
    Flask, render_template, Response, request,
    redirect, url_for, send_from_directory, flash
)
import cv2
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"   
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"mp4", "avi", "mov", "mkv"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app = Flask(__name__)
app.secret_key = "replace-with-a-secure-key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = YOLO(MODEL_PATH)


stream_control = {
    "running": False,
    "source": None,     
    "capture": None,    
    "label_queue": Queue(),
    "last_spoken": {},  
    "lock": threading.Lock()
}

SPEAK_COOLDOWN = 2.0  

def allowed_file(filename):
    ext = filename.lower().rsplit(".", 1)[-1]
    return "." in filename and ext in ALLOWED_EXT


def push_labels(labels):
    """Push labels into SSE queue but enforce cooldown per label."""
    now = time.time()
    with stream_control["lock"]:
        for lab in labels:
            last = stream_control["last_spoken"].get(lab, 0)
            if now - last >= SPEAK_COOLDOWN:
                stream_control["last_spoken"][lab] = now
                stream_control["label_queue"].put(lab)


def gen_frames(source):
  
    if source == "live":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Failed to open capture:", source)
        return

    
    with stream_control["lock"]:
        stream_control["capture"] = cap

    try:
        while True:
            with stream_control["lock"]:
                if not stream_control["running"]:
                    break
            ret, frame = cap.read()
            if not ret:
                break

            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, verbose=False, imgsz=640)[0]

            labels_in_frame = []
            
            if hasattr(results, "boxes") and len(results.boxes) > 0:
                for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    label = model.names[int(cls)]
                    confidence = float(conf)
                    labels_in_frame.append(label)

                   
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    txt = f"{label} {confidence:.2f}"
                    cv2.putText(frame, txt, (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

          
            unique_labels = list(dict.fromkeys(labels_in_frame))
            if unique_labels:
                push_labels(unique_labels)

            
            ret2, jpeg = cv2.imencode('.jpg', frame)
            if not ret2:
                continue
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        
        with stream_control["lock"]:
            if stream_control["capture"] is not None:
                stream_control["capture"].release()
                stream_control["capture"] = None
            stream_control["running"] = False
            
            try:
                while True:
                    stream_control["label_queue"].get_nowait()
            except Empty:
                pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/theory")
def theory():
    return render_template("theory.html")


@app.route("/detection")
def detection_menu():
    return render_template("detection_menu.html")


@app.route("/upload_detect", methods=["GET", "POST"])
def upload_detect():
    
    if request.method == "POST":
        if "video_file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["video_file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = str(int(time.time())) + "_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return redirect(url_for("play_uploaded", filename=filename))
        else:
            flash("Invalid file type")
            return redirect(request.url)
    return render_template("upload_detect.html")


@app.route("/play_uploaded/<filename>")
def play_uploaded(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash("Uploaded file not found")
        return redirect(url_for("upload_detect"))
    return render_template("play_uploaded.html", filename=filename)


@app.route("/live_detect")
def live_detect():
    return render_template("live_detect.html")


@app.route("/start_stream", methods=["POST"])
def start_stream():
   
    src_type = request.form.get("source", "live")
    filename = request.form.get("filename", "")
    if src_type == "uploaded":
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return {"ok": False, "error": "file not found"}, 400
        source = filepath
    else:
        source = "live"

    with stream_control["lock"]:
       
        stream_control["running"] = True
        stream_control["source"] = source
        
        stream_control["last_spoken"] = {}
        
        try:
            while True:
                stream_control["label_queue"].get_nowait()
        except Empty:
            pass

    return {"ok": True}


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    with stream_control["lock"]:
        stream_control["running"] = False
        
        if stream_control["capture"] is not None:
            try:
                stream_control["capture"].release()
            except Exception:
                pass
            stream_control["capture"] = None
    return {"ok": True}


@app.route("/video_feed")
def video_feed():
    
    source = request.args.get("source", "live")
    filename = request.args.get("filename", "")
    if source == "uploaded":
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return "File not found", 404
        source_arg = file_path
    else:
        source_arg = "live"

    
    with stream_control["lock"]:
        if not stream_control["running"]:
            return "Stream not started", 400

    return Response(gen_frames(source_arg),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/labels_stream")
def labels_stream():
   
    def event_generator():
        q = stream_control["label_queue"]
        while True:
            with stream_control["lock"]:
                if not stream_control["running"]:
                    break
            try:
                label = q.get(timeout=0.5)
            except Empty:
                continue
            
            yield f"data: {label}\n\n"
        
        yield "data: __STREAM_STOP__\n\n"

    return Response(event_generator(), mimetype='text/event-stream')



@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    # run flask
    app.run()
