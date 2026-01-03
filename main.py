import os
import time
import subprocess
import sys
import RPi.GPIO as GPIO 

# --- 1. INSTANT FEEDBACK HELPER ---
def quick_speak(text):
    try:
        subprocess.Popen(["espeak", "-s", "160", text], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass

print("\n[BOOT] System detected. Starting Headless Mode...")
quick_speak("System starting headless.")

# --- 2. IMPORTS ---
print("[BOOT] Loading OpenCV (Camera)...")
try:
    import cv2
    from picamera2 import Picamera2
except ImportError:
    quick_speak("Error. Camera library missing.")
    sys.exit(1)

print("[BOOT] Loading AI Brain...")
import torch
import numpy as np
import hydra
from concurrent.futures import ThreadPoolExecutor

# --- 3. HARDWARE TUNING ---
os.environ["OMP_NUM_THREADS"] = "3" 
os.environ["MKL_NUM_THREADS"] = "3"

try:
    from llama_cpp import Llama
except ImportError:
    print("[ERROR] llama-cpp missing.")
    sys.exit(1)

from pipelines.pipeline import InferencePipeline

# --- 4. MAIN CLASS ---
class Chaplin:
    def __init__(self):
        # --- GPIO SETUP ---
        GPIO.setmode(GPIO.BCM)
        self.BTN_REC = 17   
        self.BTN_QUIT = 27  
        
        GPIO.setup(self.BTN_REC, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.BTN_QUIT, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        GPIO.add_event_detect(self.BTN_REC, GPIO.FALLING, 
                              callback=self.toggle_record_callback, bouncetime=300)
        GPIO.add_event_detect(self.BTN_QUIT, GPIO.FALLING, 
                              callback=self.quit_callback, bouncetime=300)

        # --- CONFIGURATION ---
        self.fps = 25 
        self.width = 640 
        self.height = 480
        self.output_prefix = "pi_cam_rec"
        self.use_llm = True 
        
        self.recording = False
        self.shutdown_flag = False
        self.executor = ThreadPoolExecutor(max_workers=1)

        # --- LOCAL LLM LOAD ---
        self.model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        
        print("[SUCCESS] Libraries loaded.")
        self.speak_text("Libraries loaded.", wait=True)

        if self.use_llm:
            if not os.path.exists(self.model_path):
                self.speak_text("Error. AI model missing.")
                sys.exit(1)

            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=256,
                n_threads=3,
                verbose=False
            )
        else:
            self.llm = None

    # --- GPIO CALLBACKS ---
    def toggle_record_callback(self, channel):
        self.recording = not self.recording
        if self.recording:
            print("\n[REC] Started")
            self.play_tone(880, 0.15) 
        else:
            print("\n[REC] Stopped")
            self.play_tone(440, 0.15)

    def quit_callback(self, channel):
        print("\n[GPIO] Quit button pressed.")
        self.shutdown_flag = True

    def speak_text(self, text, wait=False):
        if not text: return
        cmd = ["espeak", "-s", "160", text]
        if wait:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def play_tone(self, frequency, duration):
        try:
            subprocess.Popen(
                ["play", "-n", "-q", "synth", str(duration), "sin", str(frequency)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except:
            pass

    def refine_with_local_llm(self, raw_text):
        if not self.use_llm or not raw_text: return raw_text
        messages = [{"role": "system", "content": "Correct grammar. Output ONLY text."},
                    {"role": "user", "content": raw_text}]
        try:
            output = self.llm.create_chat_completion(messages=messages, max_tokens=60, temperature=0.1)
            return output['choices'][0]['message']['content'].strip().strip('"')
        except:
            return raw_text

    def perform_inference(self, video_path):
        if not os.path.exists(video_path): return
        try:
            print("\n[PROCESSING] VSR Inference...")
            self.speak_text("Processing", wait=False)
            
            raw_output = self.vsr_model(video_path)
            print(f"[RESULT] Raw: '{raw_output}'")

            final_output = self.refine_with_local_llm(raw_output)
            print(f"[SPEAKING] {final_output}")
            self.speak_text(final_output, wait=False)
            return {"output": final_output, "video_path": video_path}
        except Exception as e:
            print(f"[ERROR] {e}")
            return {"output": "", "video_path": video_path}

    def start_webcam(self):
        print("[INIT] Starting Camera...")
        try:
            picam2 = Picamera2()
            config = picam2.create_video_configuration(main={"size": (self.width, self.height), "format": "BGR888"})
            picam2.configure(config)
            picam2.start()
        except Exception as e:
            self.speak_text("Camera error")
            return

        last_frame_time = time.time()
        frame_interval = 1.0 / self.fps
        futures = []
        out = None
        frame_count = 0
        output_path = ""

        print("\n[READY] System Ready (Headless).")
        self.speak_text("System ready.", wait=True)

        while True:
            # OPTIMIZATION: Removed cv2.waitKey(1) and imshow
            # Just a tiny sleep to prevent CPU spinning when not recording
            time.sleep(0.005)
            
            if self.shutdown_flag:
                self.speak_text("Shutting down.", wait=True)
                break
            
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                frame = picam2.capture_array()
                if frame is not None:
                    if self.recording:
                        # Init Writer only when needed
                        if out is None:
                            output_path = f"{self.output_prefix}_{time.time_ns()}.mp4"
                            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height))
                        
                        # RAW WRITE: No drawing, no copying. Fastest possible path.
                        out.write(frame)
                        last_frame_time = current_time
                        frame_count += 1
                        
                        # Audio "Heartbeat" every 10 seconds so you know it's working
                        if frame_count % (self.fps * 10) == 0:
                            print(f"[REC] {frame_count / self.fps:.0f}s")
                            
                    else:
                        # Stop Logic
                        if out: 
                            out.release(); out = None
                            # Only process if clip is > 1.0 second
                            if frame_count > self.fps:
                                futures.append(self.executor.submit(self.perform_inference, output_path))
                            else:
                                if os.path.exists(output_path): os.remove(output_path)
                            frame_count = 0

            # Cleanup Threads
            for fut in futures:
                if fut.done():
                    result = fut.result()
                    if result and os.path.exists(result["video_path"]): os.remove(result["video_path"])
                    futures.remove(fut)
                else:
                    break

        picam2.stop()
        if out: out.release()
        GPIO.cleanup() 
        self.executor.shutdown(wait=False)

@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    chaplin = Chaplin()
    
    # HEAVY LOADING
    torch.set_num_threads(3)
    torch.backends.quantized.engine = 'qnnpack'
    device = torch.device("cpu")
    
    try:
        print("[INIT] Loading VSR Pipeline...")
        chaplin.vsr_model = InferencePipeline(
            cfg.config_filename,
            device=device,
            detector=cfg.detector,
            face_track=True,
        )
        if hasattr(chaplin.vsr_model, 'model'):
            chaplin.vsr_model.model = torch.quantization.quantize_dynamic(
                chaplin.vsr_model.model, {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, dtype=torch.qint8
            )
        chaplin.start_webcam()
    except Exception as e:
        print(f"[FATAL] {e}")
        GPIO.cleanup()

if __name__ == "__main__":
    main()