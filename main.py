import cv2
import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from skimage import img_as_ubyte
# import imutils


import torch
# import torch.nn as nn
import cv2
import segmentation_models_pytorch as smp

def load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load("/home/river2000/monocular_depth_estimation/model1.pth"))
    model.eval()
    return model

def preprocess_image(frame):
    # Preprocess the frame to match the input size and type expected by the model
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame = frame / 255.0  # Normalize to [0, 1]
    frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0)  # Convert to (1, 3, H, W)
    return frame

def display_image(pred):
    # Display the prediction
    pred = (pred - pred.min()) / (pred.max() - pred.min())  # Normalize to [0, 1]
    pred = (pred * 255).astype("uint8")  # Convert to [0, 255]
    cv2.imshow("Prediction", pred)

def main():
    loaded_model = load_model()
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        try:
            (check, frame) = capture.read()
            frame = cv2.flip(frame, 1)
            cv2.imshow("cam_feed", frame)

            # Preprocess the frame
            preprocessed_frame = preprocess_image(frame)

            # Inference
            with torch.no_grad():
                pred = loaded_model(preprocessed_frame)
                pred = pred.squeeze(0).permute(1, 2, 0).numpy()  # Convert back to (H, W, C)

            # Display the prediction
            display_image(pred)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error: {e}")
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
