import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image, ImageTk
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog
import time

# For YOLOv8
from ultralytics import YOLO

# Load label mappings
with open('idx_to_label.json', 'r') as f:
    idx_to_label = json.load(f)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the breed classification model architecture
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
num_ftrs = model.fc.in_features
num_classes = len(idx_to_label)
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('cat_breed_classifier.pth', map_location=device))
model = model.to(device)
model.eval()

# Load the lightweight object detection model
detection_model = YOLO('yolov8n.pt')  # YOLOv8 nano model for faster performance

# Transformations for breed classification input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)
    return image.to(device)

class CatBreedIdentifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat Breed Identifier")
        self.video_running = False
        self.cap = None

        # Set default window size
        self.root.geometry('900x700')

        # Create frames
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, pady=20)

        self.display_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

        # Create buttons with styling
        button_style = {'width': 20, 'height': 2, 'font': ('Arial', 14), 'bg': '#4CAF50', 'fg': 'black'}
        self.video_button = tk.Button(self.button_frame, text="Video", command=self.run_video, **button_style)
        self.video_button.pack(pady=10)

        self.upload_button = tk.Button(self.button_frame, text="Upload", command=self.upload_image, **button_style)
        self.upload_button.pack(pady=10)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset_app, **button_style)
        # Don't pack the reset button initially

        # Create a label to display the video frames or images
        self.display_label = tk.Label(self.display_frame, bd=2, relief=tk.SUNKEN)
        # Initially hide the display label
        self.display_label.pack_forget()

    def run_video(self):
        if not self.video_running:
            # Hide buttons
            self.video_button.pack_forget()
            self.upload_button.pack_forget()

            # Show reset button
            self.reset_button.pack(pady=10)

            self.video_running = True
            self.cap = cv2.VideoCapture(0)
            # Show the display label
            self.display_label.pack(fill=tk.BOTH, expand=True)
            self.process_video()

    def process_video(self):
        if self.video_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize frame to fit display area
                frame_resized = cv2.resize(frame, (900, 600))

                # Perform object detection
                results = detection_model(frame_resized, verbose=False)

                for result in results:
                    boxes = result.boxes
                    names = detection_model.names  # Class names

                    for box in boxes:
                        cls_id = int(box.cls[0])  # Class ID
                        conf = box.conf[0].item()  # Confidence score
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)  # Bounding box coordinates

                        # Check if the detected object is a cat (class ID 15 in COCO for YOLOv8)
                        if cls_id == 15 and conf > 0.5:
                            x1, y1, x2, y2 = xyxy

                            # Draw the bounding box on the frame
                            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

                            # Crop the detected cat region for breed classification
                            cat_region = frame_resized[y1:y2, x1:x2]

                            if cat_region.size != 0:
                                input_tensor = preprocess_frame(cat_region)

                                # Perform breed classification
                                with torch.no_grad():
                                    outputs = model(input_tensor)
                                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                                # Get top N predictions
                                topk = 3  # Number of top predictions to display
                                confidences, indices = torch.topk(probabilities, topk)
                                confidences = confidences.cpu().numpy()
                                indices = indices.cpu().numpy()

                                top_predictions = []
                                for idx_pred, confidence in zip(indices, confidences):
                                    label = idx_to_label[str(idx_pred)]
                                    top_predictions.append((label, confidence))

                                # Draw a semi-transparent rectangle
                                overlay = frame_resized.copy()
                                rectangle_height = topk * 25 + 20
                                y_start = y1 - rectangle_height if y1 - rectangle_height > 0 else y1
                                cv2.rectangle(overlay, (x1, y_start), (x2, y1), (0, 0, 0), -1)
                                alpha = 0.5  # Transparency factor
                                frame_resized = cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0)

                                # Display the predictions
                                for i, (label, confidence) in enumerate(top_predictions):
                                    text = f"{label}: {confidence*100:.1f}%"
                                    y_text = y1 - 10 - i * 25
                                    if y_text < 0:
                                        y_text = y1 + 20 + i * 25  # Adjust if text goes off the screen
                                    cv2.putText(frame_resized, text, (x1 + 5, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Convert the image to RGB and then to PIL format
                cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the image in the label
                self.display_label.imgtk = imgtk
                self.display_label.configure(image=imgtk)
            else:
                # If frame not read successfully, stop video
                self.video_running = False

            # Schedule the next frame update
            self.root.after(10, self.process_video)
        else:
            # Release the capture if video is stopped
            if self.cap:
                self.cap.release()

    def upload_image(self):
        # Hide buttons
        self.video_button.pack_forget()
        self.upload_button.pack_forget()

        # Show reset button
        self.reset_button.pack(pady=10)

        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                print("Failed to load image file:", file_path)
                return

            # Resize image to fit the display area
            frame_resized = cv2.resize(image, (900, 600))

            # Perform object detection
            results = detection_model(frame_resized, verbose=False)

            for result in results:
                boxes = result.boxes
                names = detection_model.names  # Class names

                for box in boxes:
                    cls_id = int(box.cls[0])  # Class ID
                    conf = box.conf[0].item()  # Confidence score
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)  # Bounding box coordinates

                    # Check if the detected object is a cat (class ID 15 in COCO for YOLOv8)
                    if cls_id == 15 and conf > 0.5:
                        x1, y1, x2, y2 = xyxy

                        # Draw the bounding box on the frame
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

                        # Crop the detected cat region for breed classification
                        cat_region = frame_resized[y1:y2, x1:x2]

                        if cat_region.size != 0:
                            input_tensor = preprocess_frame(cat_region)

                            # Perform breed classification
                            with torch.no_grad():
                                outputs = model(input_tensor)
                                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                            # Get top N predictions
                            topk = 3  # Number of top predictions to display
                            confidences, indices = torch.topk(probabilities, topk)
                            confidences = confidences.cpu().numpy()
                            indices = indices.cpu().numpy()

                            top_predictions = []
                            for idx_pred, confidence in zip(indices, confidences):
                                label = idx_to_label[str(idx_pred)]
                                top_predictions.append((label, confidence))

                            # Draw a semi-transparent rectangle
                            overlay = frame_resized.copy()
                            rectangle_height = topk * 25 + 20
                            y_start = y1 - rectangle_height if y1 - rectangle_height > 0 else y1
                            cv2.rectangle(overlay, (x1, y_start), (x2, y1), (0, 0, 0), -1)
                            alpha = 0.5  # Transparency factor
                            frame_resized = cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0)

                            # Display the predictions
                            for i, (label, confidence) in enumerate(top_predictions):
                                text = f"{label}: {confidence*100:.1f}%"
                                y_text = y1 - 10 - i * 25
                                if y_text < 0:
                                    y_text = y1 + 20 + i * 25  # Adjust if text goes off the screen
                                cv2.putText(frame_resized, text, (x1 + 5, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Convert the image to RGB and then to PIL format
            cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the image in the label
            self.display_label.imgtk = imgtk
            self.display_label.configure(image=imgtk)

            # Show the display label
            self.display_label.pack(fill=tk.BOTH, expand=True)
        else:
            # If no file selected, reset to initial state
            self.reset_app()

    def reset_app(self):
        # Stop video if running
        if self.video_running:
            self.video_running = False
            if self.cap:
                self.cap.release()

        # Clear the display label
        self.display_label.pack_forget()
        self.display_label.imgtk = None

        # Hide reset button
        self.reset_button.pack_forget()

        # Show buttons
        self.video_button.pack(pady=10)
        self.upload_button.pack(pady=10)

    def exit_app(self):
        self.video_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

# Main GUI
root = tk.Tk()
app = CatBreedIdentifierApp(root)
root.mainloop()
