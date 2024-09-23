import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np

def init_model():
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = resnet50(pretrained=True)
    model = model.to(device)
    model.eval()

    print("Using mean and std for ImageNet dataset")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform, device

def process_frame(frame, model, transform, device):
    input_tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    _, predicted = torch.max(output, 1)
    
    cv2.putText(frame, f"Class: {predicted.item()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def init_camera():
    print("Attempting to start camera...")
    for index in range(10):  # Try indices 0 to 9
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera initialized successfully on index {index}.")
                return cap
        cap.release()
    
    print("Error: Could not open camera on any index.")
    return None

def load_classifiers():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade

def detect_face_eyes(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    return frame

def main():
    cap = init_camera()
    if cap is None:
        print("Exiting due to camera initialization failure.")
        return

    face_cascade, eye_cascade = load_classifiers()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = detect_face_eyes(frame, face_cascade, eye_cascade)
        cv2.imshow('Driver Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()