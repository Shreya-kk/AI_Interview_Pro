from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3, os, cv2, numpy as np, tensorflow as tf, pyttsx3, time, random, base64
import sounddevice as sd
import speech_recognition as sr
from collections import Counter
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import google.generativeai as genai
import fitz  # for resume PDF parsing
import io
import wave
import noisereduce as nr
from scipy.io import wavfile
import threading
import queue

app = Flask(__name__)
app.secret_key = "#"

# -----------------------------
# Gemini API setup
# -----------------------------
api_key = "API_KEY"
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# Database setup
# -----------------------------
DB_PATH = "users.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT UNIQUE,
                    contact TEXT,
                    address TEXT,
                    password TEXT)""")
    conn.commit()
    conn.close()
init_db()

# -----------------------------
# Emotion model setup
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
try:
    emotion_model = tf.keras.models.load_model("mobilenet_model.keras")
    print("✅ Emotion model loaded successfully")
except:
    print("⚠️ Emotion model not found, using demo mode")
    emotion_model = None

class_names = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# -----------------------------
# Speech Recognition Setup
# -----------------------------
recognizer = sr.Recognizer()
recognizer.pause_threshold = 2.0
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

speech_queue = queue.Queue()
is_listening = False
current_question_index = 0

# -----------------------------
# Helper functions
# -----------------------------
def extract_resume_text(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    pdf.close()
    return text

@app.route("/process-frame", methods=["POST"])
def process_frame():
    try:
        data = request.get_json()
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        emotions, processed_frame = detect_emotion(frame)
        
        return jsonify({"emotions": emotions})

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return jsonify({"error": str(e)})

def map_emotion(e):
    return {
        "happy": "Positive / Confident",
        "sad": "Nervous / Lacking confidence",
        "fear": "Anxious / Nervous",
        "angry": "Irritated / Aggressive",
        "surprise": "Alert / Reactive",
        "neutral": "Calm / Attentive",
        "disgust": "Uncomfortable / Displeased"
    }.get(e, "Calm / Attentive")

def behavior_report(emotions):
    if not emotions:
        return "No emotions detected."
    cnt = Counter(emotions)
    dom = cnt.most_common(1)[0][0]
    return f"Dominant emotion: {dom} | Candidate behavior: {map_emotion(dom)}"

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    emotions = []
    
    if len(faces) == 0:
        return emotions, frame
    
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (128, 128))
        
        if emotion_model is not None:
            try:
                img_array = np.expand_dims(roi_resized, axis=0).astype("float32")
                img_array = preprocess_input(img_array)
                predictions = emotion_model.predict(img_array, verbose=0)[0]
                predicted_class = np.argmax(predictions)
                emotion = class_names[predicted_class]
                emotions.append(emotion)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error in emotion prediction: {e}")
                emotions.append("neutral")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, "Error", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            emotions.append("neutral")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    return emotions, frame

def background_listener():
    global is_listening, current_question_index
    
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)

        while is_listening:
            try:
                print("Listening...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=10)
                
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    speech_queue.put({
                        "type": "speech",
                        "text": text,
                        "question_index": current_question_index
                    })
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    speech_queue.put({
                        "type": "silence",
                        "question_index": current_question_index
                    })
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    
            except Exception as e:
                print(f"Error in background listener: {e}")
                if is_listening:
                    time.sleep(0.1)

def start_speech_recognition():
    global is_listening, speech_thread
    is_listening = True
    speech_thread = threading.Thread(target=background_listener)
    speech_thread.daemon = True
    speech_thread.start()
    print("Speech recognition started")

def stop_speech_recognition():
    global is_listening
    is_listening = False
    print("Speech recognition stopped")

def generate_resume_question(resume_text, asked_questions=None):
    if asked_questions is None:
        asked_questions = []
    
    prompt = f"""
    Based on this candidate's resume:
    
    {resume_text}
    
    Generate ONE short, direct technical interview question that matches 
    the candidate's skills, projects, or experience. 
    Keep it simple like 'What is Python?', 'Explain OOP?', or 'Tell me about your project in machine learning'. 
    Do not explain. Only give the one-line question.
    
    Avoid these previous questions: {asked_questions}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        question = response.text.strip().split("\n")[0]
        
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        
        return question
    except Exception as e:
        print(f"Error generating question: {e}")
        return "Can you explain one of your projects?"

def generate_direct_question():
    technical_topics = [
        "Python programming", "Object-Oriented Programming", "Data Structures", 
        "Algorithms", "Database systems", "Machine Learning", "Web development",
        "Software engineering principles", "System design", "Cloud computing"
    ]
    
    topic = random.choice(technical_topics)
    prompt = f"Generate ONE short technical interview question about {topic}. Keep it simple and direct."
    
    try:
        response = gemini_model.generate_content(prompt)
        question = response.text.strip().split("\n")[0]
        
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
            
        return question
    except Exception as e:
        print(f"Error generating direct question: {e}")
        return "Explain polymorphism in Object-Oriented Programming."

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template("main.html")

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        contact = request.form['contact']
        address = request.form['address']
        password = request.form['password']
        
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        try:
            cur.execute("INSERT INTO users (name, email, contact, address, password) VALUES (?, ?, ?, ?, ?)",
                        (name, email, contact, address, password))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists. Please use a different email.", "danger")
        except Exception as e:
            flash(f"Registration error: {str(e)}", "danger")
        finally:
            conn.close()
            
    return render_template("register.html")

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        user = cur.fetchone()
        conn.close()
        
        if user:
            session['user'] = user[1]
            session['user_id'] = user[0]
            session['user_email'] = user[2]
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password", "danger")
            
    return render_template("login.html")

@app.route('/dashboard')
def dashboard():
    if "user" not in session:
        flash("Please login first", "warning")
        return redirect(url_for("login"))
    
    return render_template("dashboard.html", user=session["user"])

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out", "info")
    return redirect(url_for("home"))

@app.route('/take-test', methods=['GET','POST'])
def take_test():
    if "user" not in session:
        flash("Please login first", "warning")
        return redirect(url_for("login"))
    
    if request.method == "POST":
        mode = request.form.get("mode")
        
        if mode == "resume":
            if 'resume' not in request.files:
                flash("Please upload a resume file", "danger")
                return redirect(request.url)
                
            resume_file = request.files['resume']
            
            if resume_file.filename == '':
                flash("No file selected", "danger")
                return redirect(request.url)
                
            if resume_file and resume_file.filename.endswith('.pdf'):
                try:
                    resume_text = extract_resume_text(resume_file)
                    session['resume_text'] = resume_text
                    session['interview_mode'] = 'resume'
                    flash("Resume processed successfully!", "success")
                    return redirect(url_for("interview"))
                except Exception as e:
                    flash(f"Error processing resume: {str(e)}", "danger")
                    return redirect(request.url)
            else:
                flash("Please upload a PDF file", "danger")
                return redirect(request.url)
                
        elif mode == "direct":
            session['interview_mode'] = 'direct'
            return redirect(url_for("interview"))
    
    return render_template("take_test.html")

@app.route('/interview')
def interview():
    if "user" not in session:
        flash("Please login first", "warning")
        return redirect(url_for("login"))
        
    if 'interview_mode' not in session:
        flash("Please select an interview mode first", "warning")
        return redirect(url_for("take_test"))
    
    return render_template("interview.html", 
                         mode=session['interview_mode'],
                         user=session['user'])

@app.route('/start-interview', methods=['POST'])
def start_interview():
    if "user" not in session:
        return jsonify({"error": "Not authenticated"}), 401
        
    mode = session.get('interview_mode', 'direct')
    resume_text = session.get('resume_text', '')
    
    session['interview_questions'] = []
    session['interview_answers'] = []
    session['interview_emotions'] = []
    session['current_question_index'] = 0
    
    if mode == 'resume' and resume_text:
        question = "Tell me about yourself"
    else:
        question = "Tell me about your technical background and experience"
    
    session['interview_questions'].append(question)
    session.modified = True
    
    # Start speech recognition
    start_speech_recognition()
    
    # Return total_questions for the frontend
    max_questions = 3 if mode == 'resume' else 5
    return jsonify({
        "question": question, 
        "question_index": 0, 
        "mode": mode,
        "total_questions": max_questions
    })

@app.route('/next-question', methods=['POST'])
def next_question():
    if "user" not in session:
        return jsonify({"error": "Not authenticated"}), 401
        
    data = request.get_json()
    answer = data.get('answer', '')
    emotions = data.get('emotions', [])
    
    if 'interview_answers' not in session:
        session['interview_answers'] = []
    if 'interview_emotions' not in session:
        session['interview_emotions'] = []
        
    current_index = session.get('current_question_index', 0)
    
    if current_index < len(session.get('interview_questions', [])):
        if len(session['interview_answers']) <= current_index:
            session['interview_answers'].append(answer)
            session['interview_emotions'].append(emotions)
        else:
            session['interview_answers'][current_index] = answer
            session['interview_emotions'][current_index] = emotions
    
    session['current_question_index'] = current_index + 1
    current_index = session['current_question_index']
    
    mode = session.get('interview_mode', 'direct')
    max_questions = 3 if mode == 'resume' else 5
    
    if current_index >= max_questions:
        stop_speech_recognition()
        session.modified = True
        return jsonify({"status": "complete"})
    
    if mode == 'resume':
        resume_text = session.get('resume_text', '')
        asked_questions = session.get('interview_questions', [])
        question = generate_resume_question(resume_text, asked_questions)
    else:
        asked_questions = session.get('interview_questions', [])
        question = generate_direct_question()
        while question in asked_questions and len(asked_questions) > 0:
            question = generate_direct_question()
    
    session['interview_questions'].append(question)
    session.modified = True
    global current_question_index
    current_question_index = session['current_question_index']
    
    return jsonify({
        "question": question, 
        "question_index": current_index,
        "total_questions": max_questions
    })

@app.route('/check-speech', methods=['GET'])
def check_speech():
    try:
        if not speech_queue.empty():
            data = speech_queue.get_nowait()
            return jsonify(data)
        else:
            return jsonify({"type": "none"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/interview-results')
def interview_results():
    if "user" not in session:
        flash("Please login first", "warning")
        return redirect(url_for("login"))
        
    if 'interview_questions' not in session or 'interview_answers' not in session:
        flash("No interview data found. Please complete an interview first.", "warning")
        return redirect(url_for("dashboard"))
    
    questions = session.get('interview_questions', [])
    answers = session.get('interview_answers', [])
    emotions = session.get('interview_emotions', [])
    
    all_emotions = []
    for emotion_list in emotions:
        if isinstance(emotion_list, list):
            all_emotions.extend(emotion_list)
    
    behavior_analysis = behavior_report(all_emotions)
    
    session.pop('interview_questions', None)
    session.pop('interview_answers', None)
    session.pop('interview_emotions', None)
    session.pop('current_question_index', None)
    session.pop('resume_text', None)
    session.pop('interview_mode', None)
    
    stop_speech_recognition()
    
    return render_template("results.html", 
                         questions=questions, 
                         answers=answers, 
                         behavior_analysis=behavior_analysis,
                         user=session['user'])

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    app.run(debug=True, port=5000)
