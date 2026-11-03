# %%
from typing import Any

import cv2 as cv
import mediapipe as mp
import streamlit as st

__all__ = [
    "faceDetector",
    "faceMeshDetector",
    "handDetector",
    "poseDetector",
]

MPS: Any = mp.solutions  # type: ignore


# [start]____________________________________________________________
@st.cache_resource(max_entries=10, ttl=3600)
def mpSolutions(type):
    if type == "hands":
        return MPS.hands, MPS.drawing_utils
    elif type == "pose":
        return MPS.pose, MPS.drawing_utils
    elif type == "face_detection":
        return MPS.face_detection, MPS.drawing_utils
    elif type == "face_mesh":
        return MPS.face_mesh, MPS.drawing_utils
    raise ValueError(f"Unknown solution type: {type}")


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]


# [start]____________________________________________________________
@st.cache_resource(max_entries=10, ttl=3600)
class handDetector:
    def __init__(self, imageMode=False, numHands=2, solutionConfidence=0.5):
        self.imageMode = imageMode
        self.numHands = numHands
        self.detectionConfidence = solutionConfidence
        self.trackingConfidence = solutionConfidence

        self.mpHands, self.mpDraw = mpSolutions("hands")
        self.hands = self.mpHands.Hands(
            static_image_mode=self.imageMode,
            max_num_hands=self.numHands,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackingConfidence,
        )

    def findFeatures(self, imgRGB):
        self.results = self.hands.process(imgRGB)
        ih, iw = imgRGB.shape[:2]

        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(imgRGB, handLM, self.mpHands.HAND_CONNECTIONS)

                for idx, LM in enumerate(handLM.landmark):
                    x_pixels, y_pixels = int(LM.x * iw), int(LM.y * ih)
                    if idx in [0, 4, 8, 12, 16, 20]:
                        cv.circle(imgRGB, (x_pixels, y_pixels), 5, (255, 0, 255), cv.FILLED)

        return imgRGB


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]


# [start]____________________________________________________________
@st.cache_resource(max_entries=10, ttl=3600)
class poseDetector:
    def __init__(self, imageMode=False, smoothLandmarks=True, solutionConfidence=0.5):
        self.imageMode = imageMode
        self.smoothLandmarks = smoothLandmarks
        self.detectionConfidence = solutionConfidence
        self.trackingConfidence = solutionConfidence

        self.mpPose, self.mpDraw = mpSolutions("pose")
        self.pose = self.mpPose.Pose(
            static_image_mode=self.imageMode,
            smooth_landmarks=self.smoothLandmarks,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackingConfidence,
        )

    def findFeatures(self, imgRGB):
        self.results = self.pose.process(imgRGB)
        ih, iw = imgRGB.shape[:2]

        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(imgRGB, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

            for idx, LM in enumerate(self.results.pose_landmarks.landmark):
                x_pixels, y_pixels = int(LM.x * iw), int(LM.y * ih)
                cv.circle(imgRGB, (x_pixels, y_pixels), 5, (255, 0, 0), cv.FILLED)

        return imgRGB


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]


# [start]____________________________________________________________
@st.cache_resource(max_entries=10, ttl=3600)
class faceDetector:
    def __init__(self, solutionConfidence=0.5, modelSelection=1):
        self.detectionConfidence = solutionConfidence
        self.modelSelection = modelSelection

        self.mpFace, self.mpDraw = mpSolutions("face_detection")
        self.face = self.mpFace.FaceDetection(
            min_detection_confidence=self.detectionConfidence,
            model_selection=self.modelSelection,
        )

    def findFeatures(self, imgRGB):
        ih, iw = imgRGB.shape[:2]
        results = self.face.process(imgRGB)

        if results.detections:
            for idx, detection in enumerate(results.detections):
                mp_bbox = detection.location_data.relative_bounding_box
                bbox = (
                    int(mp_bbox.xmin * iw),
                    int(mp_bbox.ymin * ih),
                    int(mp_bbox.width * iw),
                    int(mp_bbox.height * ih),
                )
                score = round(detection.score[-1] * 100, ndigits=2)

                imgRGB = self.fancyDraw(imgRGB, bbox, score)

        return imgRGB

    def fancyDraw(self, img, bbox, score, thick_rect=2):
        x0, y0, w, h = bbox
        x1, y1 = x0 + w, y0 + h
        len_line = int(0.25 * w)
        thick_line = thick_rect + 3

        # bounding box
        cv.rectangle(img, bbox, (99, 19, 247), thick_rect)
        # border - top left: x0, y0
        cv.line(img, (x0, y0), (x0 + len_line, y0), (99, 19, 247), thick_line)
        cv.line(img, (x0, y0), (x0, y0 + len_line), (99, 19, 247), thick_line)
        # border - top right: x1, y0
        cv.line(img, (x1, y0), (x1 - len_line, y0), (99, 19, 247), thick_line)
        cv.line(img, (x1, y0), (x1, y0 + len_line), (99, 19, 247), thick_line)
        # border - bottom left: x0, y1
        cv.line(img, (x0, y1), (x0 + len_line, y1), (99, 19, 247), thick_line)
        cv.line(img, (x0, y1), (x0, y1 - len_line), (99, 19, 247), thick_line)
        # border - bottom left: x1, y1
        cv.line(img, (x1, y1), (x1 - len_line, y1), (99, 19, 247), thick_line)
        cv.line(img, (x1, y1), (x1, y1 - len_line), (99, 19, 247), thick_line)
        # score text
        cv.putText(
            img,
            f"{score}%",
            (bbox[0], bbox[1] - 8),
            cv.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        return img


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]


# [start]____________________________________________________________
@st.cache_resource(max_entries=10, ttl=3600)
class faceMeshDetector:
    def __init__(self, imageMode=False, numFaces=1, solutionConfidence=0.5):
        self.imageMode = imageMode
        self.numFaces = numFaces
        self.detectionConfidence = solutionConfidence
        self.trackingConfidence = solutionConfidence

        self.mpFaceMesh, self.mpDraw = mpSolutions("face_mesh")
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.imageMode,
            max_num_faces=self.numFaces,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackingConfidence,
        )
        self.landmark_drawSpecs = self.mpDraw.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 251, 251)
        )
        self.connection_drawSpecs = self.mpDraw.DrawingSpec(
            thickness=2, circle_radius=2, color=(251, 0, 251)
        )

    def findFeatures(self, imgRGB):
        ih, iw = imgRGB.shape[:2]
        results = self.faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for idx, faceLM in enumerate(results.multi_face_landmarks):
                self.mpDraw.draw_landmarks(
                    imgRGB,
                    faceLM,
                    self.mpFaceMesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.landmark_drawSpecs,
                    connection_drawing_spec=self.connection_drawSpecs,
                )

        return imgRGB


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
