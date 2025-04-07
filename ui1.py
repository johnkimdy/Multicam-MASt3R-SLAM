import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import os
import time

class ImageSequenceCapture:
    def __init__(self, folder_path, fps=3):
        self.folder_path = folder_path
        self.frame_interval = 1.0 / fps
        self.image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])
        self.current_frame = 0
        self.last_read_time = 0
        self.is_finished = False
        
    def read(self):
        # If we've reached the end, keep returning False
        if self.is_finished:
            return False, None
            
        current_time = time.time()
        if current_time - self.last_read_time < self.frame_interval:
            return False, None
            
        if self.current_frame >= len(self.image_files):
            self.is_finished = True  # Mark sequence as finished
            return False, None
            
        try:
            image_path = os.path.join(self.folder_path, self.image_files[self.current_frame])
            frame = cv2.imread(image_path)
            self.current_frame += 1
            self.last_read_time = current_time
            return True, frame
        except:
            self.is_finished = True  # Mark as finished if there's an error
            return False, None
            
    def release(self):
        pass

class VideoGrid:
    def __init__(self, root):
        self.root = root
        self.root.title("UI 1 (2x2 Multi-view Video Grid)")

        # Set window to fullscreen at startup
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        
        # Configure grid weights to make cells expand equally
        for i in range(2):
            self.root.grid_rowconfigure(i, weight=1)
            self.root.grid_columnconfigure(i, weight=1)
        
        # Image sequence folders
        self.image_folders = [
            "/Users/johnkim/Downloads/cups/cup1",  # Replace with your folder paths
            "/Users/johnkim/Downloads/cups/cup2",
            "/Users/johnkim/Downloads/cups/cup3"
        ]
        
        # Create 2x2 grid of frames containing labels
        self.video_labels = []
        self.camera_labels = []
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                frame = ttk.Frame(root)
                frame.grid(row=i, column=j, padx=2, pady=2, sticky="nsew")
                frame.grid_rowconfigure(0, weight=1)
                frame.grid_columnconfigure(0, weight=1)
                
                video_label = ttk.Label(frame)
                video_label.grid(row=0, column=0, sticky="nsew")
                self.video_labels.append(video_label)
                
                if idx < len(self.image_folders):
                    camera_label = tk.Label(
                        frame, 
                        text=f"Camera {idx + 1}",
                        background='white',
                        foreground='black',
                        padx=5,
                        pady=5,
                        font=('Arial', 10, 'bold')
                    )
                    camera_label.grid(row=0, column=0, sticky="nw", padx=10, pady=10)
                    self.camera_labels.append(camera_label)
        
        # Initialize image sequence captures
        self.caps = [ImageSequenceCapture(folder, fps=9) for folder in self.image_folders]
        
        # Bind window resize event
        self.root.bind('<Configure>', self.on_window_resize)
        # Calculate initial cell size based on screen dimensions
        self.cell_size = (screen_width // 2, screen_height // 2)
        
        # Start video streams
        self.running = True
        self.thread = threading.Thread(target=self.update_frames)
        self.thread.daemon = True
        self.thread.start()


    def on_window_resize(self, event):
       # Only update if it's a genuine window resize event (not from internal changes)
        if event.widget == self.root:
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            self.cell_size = (window_width // 2, window_height // 2)

    def update_frames(self):
        while self.running:
            # Update first 3 cells with video feeds
            for i, cap in enumerate(self.caps):
                ret, frame = cap.read()
                if ret:
                    # If video ends, restart from beginning
                    if frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    # Calculate aspect ratio
                    height, width = frame.shape[:2]
                    aspect_ratio = width / height
                    
                    # Calculate new dimensions maintaining aspect ratio
                    cell_w, cell_h = self.cell_size
                    if cell_w / cell_h > aspect_ratio:
                        new_width = int(cell_h * aspect_ratio)
                        new_height = cell_h
                    else:
                        new_width = cell_w
                        new_height = int(cell_w / aspect_ratio)
                    
                    # Resize frame maintaining aspect ratio
                    frame = cv2.resize(frame, (new_width, new_height))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image)
                    
                    # Update label
                    self.video_labels[i].configure(image=photo)
                    self.video_labels[i].image = photo
            
            # Add small delay to control frame rate
            self.root.after(30)

    def __del__(self):
        self.running = False
        for cap in self.caps:
            cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoGrid(root)
    root.mainloop()