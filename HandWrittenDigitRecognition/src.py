import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import threading

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.image = Image.new("L", (280, 280), 0)
        self.image_draw = ImageDraw.Draw(self.image)  # Changed variable name from draw to image_draw
        
        # Load MNIST dataset in a separate thread
        self.loading_label = ttk.Label(root, text="Loading MNIST dataset and training model...")
        self.loading_label.pack(pady=20)
        
        self.progress = ttk.Progressbar(root, mode='indeterminate')
        self.progress.pack(pady=10)
        self.progress.start()
        
        threading.Thread(target=self.load_and_train_model, daemon=True).start()
        
    def load_and_train_model(self):
        # Load MNIST dataset from OpenML
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        X, y = mnist.data, mnist.target
        
        # Preprocess the data
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train K-Nearest Neighbors classifier
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.knn.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        # Update UI when model is ready
        self.root.after(0, self.setup_ui)
    
    def setup_ui(self):
        # Remove loading widgets
        self.loading_label.pack_forget()
        self.progress.pack_forget()
        
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Drawing canvas
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(left_frame, width=280, height=280, bg='black')
        self.canvas.pack(pady=10)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Drawing controls
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(pady=10)
        
        clear_btn = ttk.Button(controls_frame, text="Clear", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        predict_btn = ttk.Button(controls_frame, text="Predict", command=self.predict_digit)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Results
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20)
        
        # Model info
        info_frame = ttk.LabelFrame(right_frame, text="Model Information")
        info_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(info_frame, text="Algorithm:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(info_frame, text="K-Nearest Neighbors (k=3)").grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(info_frame, text="Accuracy:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(info_frame, text=f"{self.accuracy:.2%}").grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Prediction results
        result_frame = ttk.LabelFrame(right_frame, text="Prediction Results")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.prediction_label = ttk.Label(result_frame, text="Draw a digit and click Predict", 
                                         font=('Helvetica', 24))
        self.prediction_label.pack(pady=20)
        
        self.confidence_label = ttk.Label(result_frame, text="")
        self.confidence_label.pack(pady=10)
        
        # Sample digits
        sample_frame = ttk.LabelFrame(right_frame, text="Try These Samples")
        sample_frame.pack(fill=tk.BOTH, pady=10)
        
        sample_btns = []
        for i in range(10):
            btn = ttk.Button(sample_frame, text=str(i), width=2, 
                            command=lambda x=i: self.load_sample(x))
            btn.pack(side=tk.LEFT, padx=5)
            sample_btns.append(btn)
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
    
    def draw(self, event):
        if self.drawing:
            # Draw on canvas
            self.canvas.create_line((self.last_x, self.last_y, event.x, event.y), 
                                  fill='white', width=15, capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # Draw on PIL image
            self.image_draw.line([self.last_x, self.last_y, event.x, event.y], 
                              fill=255, width=15)
            
            self.last_x, self.last_y = event.x, event.y
    
    def stop_drawing(self, event):
        self.drawing = False
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.image_draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit and click Predict")
        self.confidence_label.config(text="")
    
    def preprocess_image(self):
        # Resize to 28x28 (MNIST size)
        img = self.image.resize((28, 28), Image.LANCZOS)
        
        # Convert to numpy array and flatten
        img_array = np.array(img).reshape(1, -1)
        
        # Scale using the same scaler used for training
        return self.scaler.transform(img_array)
    
    def predict_digit(self):
        try:
            # Preprocess the drawn image
            processed_img = self.preprocess_image()
            
            # Predict
            prediction = self.knn.predict(processed_img)[0]
            probabilities = self.knn.predict_proba(processed_img)[0]
            confidence = probabilities[int(prediction)]
            
            # Update UI
            self.prediction_label.config(text=f"Prediction: {prediction}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def load_sample(self, digit):
        # Clear current drawing
        self.clear_canvas()
        
        # Load a sample digit from MNIST
        sample_indices = np.where(self.knn.classes_ == str(digit))[0]
        if len(sample_indices) > 0:
            sample = self.knn._fit_X[sample_indices[0]].reshape(28, 28)
            sample = (sample * 255).astype(np.uint8)
            
            # Resize to fit our canvas
            sample_img = Image.fromarray(sample).resize((280, 280), Image.NEAREST)
            
            # Display on canvas
            self.tk_sample = ImageTk.PhotoImage(sample_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_sample)
            
            # Update PIL image
            self.image = sample_img.copy()
            self.image_draw = ImageDraw.Draw(self.image)
            
            # Predict automatically
            self.predict_digit()
        else:
            messagebox.showwarning("Warning", f"No sample found for digit {digit}")

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()