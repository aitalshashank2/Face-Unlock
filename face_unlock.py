from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import h5py
import pickle
import face_recognition
from PIL import Image, ImageDraw
import os

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the (optional) video file.")
args = vars(ap.parse_args())

with open("trained_knn_model.clf", "rb") as f:
    knn_clf = pickle.load(f)

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, image1) = camera.read()
    if args.get("video") and not grabbed:
        break

    image = image1[:, :, ::-1]

    X_face_locations = face_recognition.face_locations(image)

    if len(X_face_locations) != 0:
        face_encodings = face_recognition.face_encodings(image, known_face_locations=X_face_locations)

        print(np.array(face_encodings).shape)
        closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(X_face_locations))]

        predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(face_encodings), X_face_locations, are_matches)]

        for name, (top, right, bottom, left) in predictions:
            if name not in "unknown":
                os.popen("gnome-screensaver-command -d && xdotool key Return")

            cv2.rectangle(image1, (left, bottom), (right, top), (0, 255, 0), 2)

            cv2.putText(image1, f"{name}", (left-10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    else:
        os.popen('gnome-screensaver-command -a')

    cv2.imshow("Output image", image1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
