import os
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import hydra
import torch
import numpy as np

# --- 1. OPTIMIZATION: LIMIT THREADS ---
# Keeps the OS responsive while speaking
os.environ["OMP_NUM_THREADS"] = "3" 
os.environ["MKL_NUM_THREADS"] = "3"

try:
    from llama_cpp import Llama
except ImportError:
    print("[ERROR] llama-cpp-python not found.")
    exit(1)

try:
    from picamera2 import Picamera2
except ImportError:
    print("[ERROR] picamera2 not found.")
    exit(1)

from pipelines.pipeline import InferencePipeline

class Chaplin:
    def __init__(self):
        # --- VOICE HELPERS INITIALIZED FIRST ---
        self.last_spoken_time = 0
        
        # Announce Startup
        print("[VOICE] Initializing  system...")
        self.speak_text("Initializing  system", wait=True)

        self.vsr_model = None
        
        # --- CONFIGURATION ---
        self.fps = 25 
        self.width = 640 
        self.height = 480
        self.output_prefix = "pi_cam_rec"
        self.use_llm = True 
        
        # --- LOCAL LLM LOAD ---
        self.model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        
        if self.use_llm:
            if not os.path.exists(self.model_path):
                self.speak_text("Error. Language model not found.", wait=True)
                print(f"[ERROR] Model not found at {self.model_path}")
                exit(1)

            print("[INIT] Loading local LLM...")
            self.speak_text("Loading language model.", wait=True) # Voice Instruction
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=256,
                n_threads=3,
                verbose=False
            )
            print("[INIT] Local LLM Loaded.")
        else:
            self.llm = None

        self.recording = False
        self.executor = ThreadPoolExecutor(max_workers=1)

    def speak_text(self, text, wait=False):
        """
        wait=True: Pauses code until speaking is done (Good for startup instructions)
        wait=False: Speaks in background (Good for results)
        """
        if not text: return
        try:
            # -s 160 = speed (slightly faster than default)
            cmd = ["espeak", "-s", "160", text]
            
            if wait:
                # Blocks execution so heavy CPU load doesn't stutter the audio
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Runs in background
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[ERROR] Audio playback failed: {e}")

    def play_tone(self, frequency, duration):
        try:
            subprocess.Popen(
                ["play", "-n", "-q", "synth", str(duration), "sin", str(frequency)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except:
            pass

    def refine_with_local_llm(self, raw_text):
        if not self.use_llm or not raw_text:
            return raw_text

        messages = [
            {
                "role": "system",
                "content": "Correct grammar and homophenes. Output ONLY the corrected text."
            },
            {
                "role": "user",
                "content": f"{raw_text}"
            }
        ]

        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=60,
                temperature=0.1
            )
            return output['choices'][0]['message']['content'].strip().strip('"')
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return raw_text

    def perform_inference(self, video_path):
        if not os.path.exists(video_path): return

        try:
            print("\n[PROCESSING] VSR Inference...")
            #self.speak_text("Processing", wait=False) # Brief feedback
            
            start_t = time.time()
            raw_output = self.vsr_model(video_path)
            vsr_time = time.time() - start_t
            
            print(f"[RESULT] Raw: '{raw_output}' ({vsr_time:.2f}s)")

            final_output = self.refine_with_local_llm(raw_output)
            
            if final_output != raw_output:
                print(f"[LLM FIX] -> '{final_output}'")

            print(f"[SPEAKING] {final_output}")
            self.speak_text(final_output, wait=False)
            
            return {"output": final_output, "video_path": video_path}
            
        except Exception as e:
            print(f"[ERROR] {e}")
            self.speak_text("Error in processing", wait=False)
            return {"output": "", "video_path": video_path}

    def start_webcam(self):
        print("[INIT] Starting Camera...")
        self.speak_text("Starting camera module.", wait=True) # Voice Instruction

        try:
            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"} 
            )
            picam2.configure(config)
            picam2.start()
        except Exception as e:
            self.speak_text("Camera failed to start.", wait=True)
            print(f"[ERROR] Camera failed: {e}")
            return

        last_frame_time = time.time()
        frame_interval = 1.0 / self.fps
        
        futures = []
        output_path = ""
        out = None
        frame_count = 0

        print("\n[READY] System Ready. Press 'R' to Record, 'Q' to Quit")
        # Final "Ready" instruction
        self.speak_text("System ready. Press R to record.", wait=True)

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                print("\n[SHUTDOWN] Exiting...")
                self.speak_text("Shutting down. Goodbye.", wait=True)
                break
            
            if key == ord("r"):
                self.recording = not self.recording
                if self.recording:
                    print("\n[REC] Started")
                    self.play_tone(880, 0.15) 
                else:
                    print("\n[REC] Stopped")
                    self.play_tone(440, 0.15)

            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                frame = picam2.capture_array()
                
                if frame is not None:
                    display_frame = frame 
                    
                    if self.recording:
                        if out is None:
                            output_path = f"{self.output_prefix}_{time.time_ns()}.mp4"
                            out = cv2.VideoWriter(
                                output_path, 
                                cv2.VideoWriter_fourcc(*"mp4v"), 
                                self.fps, 
                                (self.width, self.height)
                            )
                        
                        out.write(frame)
                        last_frame_time = current_time
                        frame_count += 1
                        
                        if frame_count % 5 == 0:
                            display_frame = frame.copy()
                            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)

                    elif not self.recording and frame_count > 0:
                        if out: out.release(); out = None
                        
                        duration = frame_count / self.fps
                        if duration >= 0.5:
                            print(f"[JOB] Processing {duration:.1f}s clip...")
                            futures.append(self.executor.submit(self.perform_inference, output_path))
                        else:
                            if os.path.exists(output_path): os.remove(output_path)
                        frame_count = 0

                    cv2.imshow("Chaplin Pi", display_frame)

            for fut in futures:
                if fut.done():
                    result = fut.result()
                    if result and os.path.exists(result["video_path"]): 
                        os.remove(result["video_path"])
                    futures.remove(fut)
                else:
                    break

        picam2.stop()
        if out: out.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=False)

@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    chaplin = Chaplin()
    
    # --- VOICE INSTRUCTION: VSR LOADING ---
    print("[INIT] Loading VSR Model...")
    chaplin.speak_text("Loading visual speech recognition model.", wait=True)
    
    # OPTIMIZATION
    torch.set_num_threads(3)
    torch.backends.quantized.engine = 'qnnpack'
    device = torch.device("cpu")
    
    try:
        chaplin.vsr_model = InferencePipeline(
            cfg.config_filename,
            device=device,
            detector=cfg.detector,
            face_track=True,
        )
        
        # QUANTIZATION
        print("[INIT] Quantizing model...")
        if hasattr(chaplin.vsr_model, 'model'):
            chaplin.vsr_model.model = torch.quantization.quantize_dynamic(
                chaplin.vsr_model.model, 
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, 
                dtype=torch.qint8
            )
        
        chaplin.start_webcam()

    except Exception as e:
        chaplin.speak_text("Fatal error. System stopping.", wait=True)
        print(f"[FATAL] {e}")

if __name__ == "__main__":
    main()