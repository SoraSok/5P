# Importing Libraries
import numpy as np
import math
import cv2
import os, sys
import traceback
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
# Google Gemini Pro for LLM-powered context-aware suggestions
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("Google Gemini Pro library imported successfully")
except ImportError as e:
    GEMINI_AVAILABLE = False
    print(f"Warning: Google Gemini Pro not installed. Error: {e}")
    print("Install with: pip install google-generativeai")

# ElevenLabs for high-quality text-to-speech
try:
    from elevenlabs.client import ElevenLabs
    try:
        from elevenlabs.play import play as play_audio
        PLAY_AVAILABLE = True
    except ImportError:
        try:
            from elevenlabs import play as play_module
            play_audio = play_module.play if hasattr(play_module, 'play') else None
            PLAY_AVAILABLE = play_audio is not None
        except:
            PLAY_AVAILABLE = False
            play_audio = None
    import sounddevice as sd
    try:
        import soundfile as sf
        SOUNDFILE_AVAILABLE = True
    except ImportError:
        SOUNDFILE_AVAILABLE = False
    ELEVENLABS_AVAILABLE = True
    print("ElevenLabs library imported successfully")
except ImportError as e:
    ELEVENLABS_AVAILABLE = False
    SOUNDFILE_AVAILABLE = False
    print(f"Warning: ElevenLabs not installed. Error: {e}")
    print("Text-to-speech will not be available.")

hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

offset=29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


# Application :

class Application:

    def __init__(self):
        # Try to find an available camera
        self.vs = None
        for i in range(3):  # Try camera indices 0, 1, 2
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.vs = cap
                    print(f"Camera found at index {i}")
                    break
                else:
                    cap.release()
            else:
                cap.release()
        
        if self.vs is None:
            print("ERROR: No camera found! Please connect a webcam.")
            self.vs = cv2.VideoCapture(0)  # Fallback, will show errors but won't crash
        self.current_image = None
        self.model = load_model('./cnn8grps_rad1_model.h5')
        
        # Initialize Google Gemini Pro for LLM-powered features
        self.gemini_api_key = "AIzaSyDSi4llUp_OWYaJGnKptnSWip6WQkgjSwY"  # Hardcoded API key
        self.gemini_model = None
        self.use_gemini = False
        self.conversation_history = []  # Store conversation for context
        self.gemini_request_pending = False  # Prevent multiple simultaneous requests
        self.last_word_request_time = 0  # For debouncing
        self.last_word = ""  # Track last word to avoid duplicate requests
        
        if GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.gemini_api_key)
                # Use gemini-2.5-flash for fast, context-aware suggestions
                # Alternative: 'gemini-2.5-pro' for better quality but slower
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                self.use_gemini = True
                print("Google Gemini Pro LLM enabled - using AI-powered context-aware suggestions")
            except Exception as e:
                print(f"Warning: Could not configure Gemini Pro API: {e}")
                print("Word suggestions will not be available.")
        else:
            print("Google Gemini Pro library not available. Install with: pip install google-generativeai")
            print("Word suggestions will not be available.")
        
        # Initialize text-to-speech engines
        # ElevenLabs API key (hardcoded)
        self.use_elevenlabs = False
        self.elevenlabs_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default: Rachel (neutral, clear voice)
        self.elevenlabs_api_key = "sk_485cb28dd681c4ed6d596613c86cc91c29edb9c612b18be3"  # Hardcoded API key
        self.elevenlabs_client = None
        
        if ELEVENLABS_AVAILABLE:
            try:
                # Initialize ElevenLabs client with hardcoded API key
                self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)
                self.use_elevenlabs = True
                print("ElevenLabs TTS enabled - using high-quality voice synthesis")
            except Exception as e:
                print(f"Warning: Could not configure ElevenLabs API: {e}")
                print("Text-to-speech will not be available.")
        else:
            print("ElevenLabs library not available. Install with: pip install elevenlabs")
            print("Text-to-speech will not be available.")


        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag=False
        self.next_flag=True
        self.prev_char=""
        self.count=-1
        self.ten_prev_char=[]
        for i in range(10):
            self.ten_prev_char.append(" ")


        for i in ascii_uppercase:
            self.ct[i] = 0

        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1300x700")

        self.panel = tk.Label(self.root)
        self.panel.place(x=40, y=3, width=480, height=640)

        self.panel2 = tk.Label(self.root)  # initialize image panel
        self.panel2.place(x=550, y=115, width=400, height=400)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Times New Roman", 30, "bold"))

        image1 = Image.open("signs.png")
        image1= image1.resize((500,400), Image.LANCZOS)
        test = ImageTk.PhotoImage(image1)
        label1 = tk.Label(image=test)
        label1.image = test
        label1.place(x=1000,y=110)

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=280, y=585)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=580)
        self.T1.config(text="Character :", font=("Times New Roman", 30, "bold"))

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=260, y=632)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=632)
        self.T3.config(text="Sentence :", font=("Times New Roman", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=10, y=700)
        self.T4.config(text="Suggestions :", fg="red", font=("Times New Roman", 30, "bold"))


        self.b1=tk.Button(self.root)
        self.b1.place(x=390,y=700)

        self.b2 = tk.Button(self.root)
        self.b2.place(x=590, y=700)

        self.b3 = tk.Button(self.root)
        self.b3.place(x=790, y=700)

        self.b4 = tk.Button(self.root)
        self.b4.place(x=990, y=700)

        self.speak = tk.Button(self.root)
        self.speak.place(x=1305, y=630)
        self.speak.config(text="Speak", font=("Times New Roman", 20), wraplength=100, command=self.speak_fun)

        self.clear = tk.Button(self.root)
        self.clear.place(x=1205, y=630)
        self.clear.config(text="Clear", font=("Times New Roman", 20), wraplength=100, command=self.clear_fun)

        # AI Assistant button for grammar correction and context-aware suggestions
        self.ai_assistant = tk.Button(self.root)
        self.ai_assistant.place(x=1105, y=630)
        self.ai_assistant.config(text="AI Fix", font=("Times New Roman", 20), wraplength=100, 
                                 command=self.ai_assistant_fun, bg="#4CAF50", fg="white")






        self.str = " "
        self.ccc=0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"


        self.word1=" "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "


        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok or frame is None:
                # Camera not accessible, skip this frame
                self.root.after(1, self.video_loop)
                return
            cv2image = cv2.flip(frame, 1)
            hands, cv2image = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy=np.array(cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                # #print(" --------- lmlist=",hands[1])
                hand = hands[0]
                # Check if hand is a dict (old API) or handle new format
                if isinstance(hand, dict):
                    x, y, w, h = hand['bbox']
                else:
                    # New API format - hand might be a list/tuple, skip for now
                    self.root.after(1, self.video_loop)
                    return
                
                # Check bounds to prevent out-of-bounds cropping
                frame_h, frame_w = cv2image_copy.shape[:2]
                x1 = max(0, x - offset)
                y1 = max(0, y - offset)
                x2 = min(frame_w, x + w + offset)
                y2 = min(frame_h, y + h + offset)
                
                if x2 > x1 and y2 > y1:
                    image = cv2image_copy[y1:y2, x1:x2]
                    
                    if image.size > 0:
                        white = cv2.imread("./white.jpg")
                        # img_final=img_final1=img_final2=0

                        handz, _ = hd2.findHands(image, draw=False, flipType=True)
                        # Removed verbose logging for performance
                        self.ccc += 1
                        if handz:
                            hand = handz[0]
                            self.pts = hand['lmList']
                            # x1,y1,w1,h1=hand['bbox']

                            os = ((400 - w) // 2) - 15
                            os1 = ((400 - h) // 2) - 15
                            for t in range(0, 4, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(5, 8, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(9, 12, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(13, 16, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(17, 20, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                                     3)
                            cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                                     3)
                            cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                                     (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                                     3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                                     3)

                            for i in range(21):
                                cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                            res=white
                            self.predict(res)

                            self.current_image2 = Image.fromarray(res)

                            imgtk = ImageTk.PhotoImage(image=self.current_image2)

                            self.panel2.imgtk = imgtk
                            self.panel2.config(image=imgtk)

                            self.panel3.config(text=self.current_symbol, font=("Times New Roman", 30))

                            #self.panel4.config(text=self.word, font=("Times New Roman", 30))



                            self.b1.config(text=self.word1, font=("Times New Roman", 20), wraplength=825, command=self.action1)
                            self.b2.config(text=self.word2, font=("Times New Roman", 20), wraplength=825,  command=self.action2)
                            self.b3.config(text=self.word3, font=("Times New Roman", 20), wraplength=825,  command=self.action3)
                            self.b4.config(text=self.word4, font=("Times New Roman", 20), wraplength=825,  command=self.action4)

            self.panel5.config(text=self.str, font=("Times New Roman", 30), wraplength=1025)
        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def distance(self,x,y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()


    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str=self.str[:idx_word]
        self.str=self.str+self.word2.upper()
        #self.str[idx_word:last_idx] = self.word2


    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()



    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()


    def speak_fun(self):
        """Convert the current sentence to speech using ElevenLabs"""
        text_to_speak = self.str.strip()
        
        if not text_to_speak or text_to_speak == " ":
            print("No text to speak")
            return
        
        # Use ElevenLabs if available and configured
        if self.use_elevenlabs and ELEVENLABS_AVAILABLE and self.elevenlabs_client:
            try:
                # Reduced logging for performance
                
                # Generate audio from ElevenLabs using the client API
                # Using eleven_turbo_v2_5 or eleven_multilingual_v2 which are available on free tier
                audio_stream = self.elevenlabs_client.text_to_speech.convert(
                    voice_id=self.elevenlabs_voice_id,
                    text=text_to_speak,
                    model_id="eleven_turbo_v2_5"  # Free tier compatible model (alternative: "eleven_multilingual_v2")
                )
                
                # Convert generator to bytes
                audio_bytes = b''.join(audio_stream)
                
                # Play the audio - try multiple methods for reliability
                # Method 1: Use sounddevice + soundfile (most reliable for Windows)
                playback_success = False
                if SOUNDFILE_AVAILABLE:
                    try:
                        import tempfile
                        
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        
                        # Play using soundfile and sounddevice
                        data, samplerate = sf.read(tmp_path)
                        sd.play(data, samplerate)
                        sd.wait()  # Wait until playback is finished
                        
                        # Clean up
                        os.unlink(tmp_path)
                        playback_success = True
                        # Audio played successfully
                    except Exception:
                        pass  # Try next method
                
                # Method 2: Try ElevenLabs play() function if available
                if not playback_success and PLAY_AVAILABLE and play_audio:
                    try:
                        from io import BytesIO
                        play_audio(BytesIO(audio_bytes))
                        playback_success = True
                        # Audio played successfully
                    except Exception:
                        pass  # Try next method
                
                # Method 3: Save to file and use system player as last resort
                if not playback_success:
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        
                        # Use system default player
                        import subprocess
                        import platform
                        if platform.system() == 'Windows':
                            os.startfile(tmp_path)
                        else:
                            subprocess.call(['open' if platform.system() == 'Darwin' else 'xdg-open', tmp_path])
                        # Audio opened with system player
                    except Exception:
                        pass  # Reduced verbose error logging
                
                return  # Success, exit early
                
            except Exception as e:
                print(f"ElevenLabs error: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("Please check:")
                print("1. Your API key is valid")
                print("2. You have sufficient API credits")
                print("3. Your internet connection is working")
        else:
            print("ElevenLabs is not configured properly.")
            print(f"use_elevenlabs: {self.use_elevenlabs}, ELEVENLABS_AVAILABLE: {ELEVENLABS_AVAILABLE}, client: {self.elevenlabs_client is not None}")

   

    def clear_fun(self):
        self.str=" "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def predict(self, test_image):
        white=test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0
                # print("00000")

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2
                # print("22222")

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2
                # print("22222")


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3
                print("33333c")



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3
                print("33333b")

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3
                print("33333a")

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4
                # print("44444")

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4
                # print("44444")

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
                # print("44444")

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5
                print("55555b")

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5
                print("55555a")

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5
                # print("55555")

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5
                # print("55555")

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7
                # print("77777")

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7
                # print("77777")

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7
                # print("77777")

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6
                print("666661")


        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6
                print("666662")

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6
                print("666663")

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6
                print("666664")

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6
                print("666665")

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1
                print("111111")

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
                print("111112")

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
                print("111112")

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1
                print("111113")

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1
                print("1111993")

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1
                print("1111mmm3")

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1
                print("1111140")

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1
                print("111114")

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7
                print("111114lll;;p")

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1
                print("111115")

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1
                print("111116")

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1
                print("1117")

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1=" "



        print(self.pts[4][0] < self.pts[5][0])
        if ch1 == 'E' or ch1=='Y' or ch1=='B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1="next"


        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'


        if ch1=="next" and self.prev_char!="next":
            if self.ten_prev_char[(self.count-2)%10]!="next":
                if self.ten_prev_char[(self.count-2)%10]=="Backspace":
                    self.str=self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count-2)%10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]


        if ch1=="  " and self.prev_char!="  ":
            self.str = self.str + "  "

        self.prev_char=ch1
        self.current_symbol=ch1
        self.count += 1
        self.ten_prev_char[self.count%10]=ch1

        if len(self.str.strip())!=0:
            st=self.str.rfind(" ")
            ed=len(self.str)
            word=self.str[st+1:ed]
            self.word=word
            
            # Initialize word suggestions to empty
            self.word1 = " "
            self.word2 = " "
            self.word3 = " "
            self.word4 = " "
            
            # Update with Gemini suggestions in background (non-blocking, async)
            # Use debouncing to avoid too many API calls
            current_time = time.time()
            if (len(word.strip()) != 0 and self.use_gemini and self.gemini_model and 
                word.strip() != self.last_word and 
                current_time - self.last_word_request_time > 0.5 and  # Debounce: 500ms delay
                not self.gemini_request_pending):
                self.last_word = word.strip()
                self.last_word_request_time = current_time
                # Run Gemini in background thread to avoid freezing UI
                threading.Thread(target=self.get_gemini_word_suggestions_async, 
                               args=(word, self.str), daemon=True).start()


    def get_gemini_word_suggestions_async(self, word, sentence):
        """Get context-aware word suggestions from Gemini Pro (async, non-blocking)"""
        if self.gemini_request_pending:
            return  # Skip if already processing
        
        self.gemini_request_pending = True
        try:
            self._get_gemini_word_suggestions(word, sentence)
        finally:
            self.gemini_request_pending = False
    
    def _extract_text_from_gemini_response(self, response):
        """Safely extract plain text from a Gemini response object"""
        if response is None:
            return ""
        try:
            if hasattr(response, "text") and response.text:
                return response.text
            # Try to walk candidates/parts for text content
            candidates = getattr(response, "candidates", None)
            if candidates:
                for candidate in candidates:
                    content = getattr(candidate, "content", None)
                    if not content:
                        continue
                    parts = getattr(content, "parts", None)
                    if parts:
                        texts = [getattr(part, "text", "") for part in parts if getattr(part, "text", "")]
                        if texts:
                            return " ".join(texts).strip()
            # Fallback to dict representation if available
            if hasattr(response, "to_dict"):
                resp_dict = response.to_dict()
                # Navigate common structure
                for candidate in resp_dict.get("candidates", []):
                    content = candidate.get("content", {})
                    for part in content.get("parts", []):
                        text = part.get("text")
                        if text:
                            return text.strip()
            return ""
        except Exception:
            return ""

    def _get_gemini_word_suggestions(self, word, sentence):
        """Internal method: Get context-aware word suggestions from Gemini Pro based on sentence context"""
        try:
            # Create prompt for context-aware suggestions
            prompt = f"""Given the incomplete sentence: "{sentence.strip()}"
And the current word being typed: "{word.strip()}"

Provide exactly 4 word suggestions that:
1. Could replace or complete the word "{word}" in this context
2. Are grammatically correct and make sense in the sentence
3. Are common words that fit the context
4. Are ordered from most likely to least likely

Return ONLY a comma-separated list of exactly 4 words, no explanations, no numbering, just: word1, word2, word3, word4
If you can't suggest 4 words, repeat the most likely ones."""

            response = self.gemini_model.generate_content(prompt)
            suggestions_text = response.text.strip()
            
            # Parse suggestions (handle various formats)
            # Try to extract words from the response
            suggestions = []
            # Split by comma and clean up
            parts = suggestions_text.split(',')
            for part in parts:
                cleaned = part.strip()
                # Remove common prefixes like "1.", "2.", etc.
                if '.' in cleaned and cleaned[0].isdigit():
                    cleaned = cleaned.split('.', 1)[1].strip()
                # Remove quotes
                cleaned = cleaned.strip('"\'')
                if cleaned and len(cleaned) > 0:
                    suggestions.append(cleaned)
            suggestions = suggestions[:4]  # Take first 4
            
            # Fill suggestions
            if len(suggestions) >= 4:
                self.word1 = suggestions[0] if len(suggestions[0]) <= 20 else suggestions[0][:20]
                self.word2 = suggestions[1] if len(suggestions[1]) <= 20 else suggestions[1][:20]
                self.word3 = suggestions[2] if len(suggestions[2]) <= 20 else suggestions[2][:20]
                self.word4 = suggestions[3] if len(suggestions[3]) <= 20 else suggestions[3][:20]
            elif len(suggestions) >= 3:
                self.word1 = suggestions[0] if len(suggestions[0]) <= 20 else suggestions[0][:20]
                self.word2 = suggestions[1] if len(suggestions[1]) <= 20 else suggestions[1][:20]
                self.word3 = suggestions[2] if len(suggestions[2]) <= 20 else suggestions[2][:20]
                self.word4 = " "
            elif len(suggestions) >= 2:
                self.word1 = suggestions[0] if len(suggestions[0]) <= 20 else suggestions[0][:20]
                self.word2 = suggestions[1] if len(suggestions[1]) <= 20 else suggestions[1][:20]
                self.word3 = " "
                self.word4 = " "
            elif len(suggestions) >= 1:
                self.word1 = suggestions[0] if len(suggestions[0]) <= 20 else suggestions[0][:20]
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "
            else:
                # No suggestions available
                self.word1 = self.word2 = self.word3 = self.word4 = " "
            
            # Reduced logging for performance
        except Exception:
            pass  # Silent failure - no suggestions available
            self.word1 = self.word2 = self.word3 = self.word4 = " "
    

    def _update_suggestion_buttons_from_suggestions(self, w1, w2, w3, w4):
        """Update suggestion buttons in GUI from async Gemini results (thread-safe)"""
        try:
            self.word1, self.word2, self.word3, self.word4 = w1, w2, w3, w4
            if len(w1.strip()) > 0 or len(w2.strip()) > 0 or len(w3.strip()) > 0 or len(w4.strip()) > 0:
                self.b1.config(text=w1, font=("Times New Roman", 20), wraplength=825, command=self.action1)
                self.b2.config(text=w2, font=("Times New Roman", 20), wraplength=825, command=self.action2)
                self.b3.config(text=w3, font=("Times New Roman", 20), wraplength=825, command=self.action3)
                self.b4.config(text=w4, font=("Times New Roman", 20), wraplength=825, command=self.action4)
        except:
            pass  # Fail silently if buttons not ready

    def ai_assistant_fun(self):
        """AI Assistant function: Grammar correction and sentence completion using Gemini Pro"""
        if not self.use_gemini or not self.gemini_model:
            print("Gemini Pro not available. Cannot use AI Assistant.")
            return
        
        current_text = self.str.strip()
        if not current_text:
            print("No text to process with AI Assistant.")
            return
        
        try:
            # Create prompt for grammar correction and improvement
            prompt = f"""You are an AI assistant helping a user communicate via sign language translation. 
The user typed this sentence (which may have spelling/grammar errors from sign-to-text translation):
"{current_text}"

Please:
1. Correct any spelling and grammar errors
2. Keep the meaning and intent the same
3. Make it natural and readable
4. Return ONLY the corrected sentence, nothing else.

If the sentence is already correct, return it as-is."""

            response = self.gemini_model.generate_content(prompt)
            corrected_text = self._extract_text_from_gemini_response(response).strip()
            if not corrected_text:
                print("Gemini AI Fix returned no text. Sentence left unchanged.")
                return
            
            # Remove quotes if present
            if corrected_text.startswith('"') and corrected_text.endswith('"'):
                corrected_text = corrected_text[1:-1]
            elif corrected_text.startswith("'") and corrected_text.endswith("'"):
                corrected_text = corrected_text[1:-1]
            
            # Update the sentence
            self.str = " " + corrected_text
            
            # Update conversation history
            self.conversation_history.append({
                "original": current_text,
                "corrected": corrected_text
            })
            
            # AI Assistant correction applied
            
            # Update the GUI display
            self.panel5.config(text=self.str.strip())
            
        except Exception as e:
            print(f"Error in AI Assistant: {e}")
            import traceback
            traceback.print_exc()

    def destructor(self):

        print("Closing Application...")
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

(Application()).root.mainloop()
