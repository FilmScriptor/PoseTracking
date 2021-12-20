import cv2
import mediapipe as mp


class poseDectector():
    def __init__(self, mode=False, complexity=1, smooth=True, enableSeg=False, smoothSeg=True, dectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionCon = dectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.enableSeg, self.smoothSeg, self.detectionCon, self.trackCon)

        #draw landmarks
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert img to RGB
        self.results = self.pose.process(imgRGB)

        #check if it detect it or not
        if self.results.pose_landmarks:
            PoseLms = self.results.pose_landmarks
            self.mpDraw.draw_landmarks(img, PoseLms, self.mpPose.POSE_CONNECTIONS,
                                       self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                       self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        return img

    def findPosition(self, img, draw=True):
        lmList = []

        if self.results.pose_landmarks:
            #print out land marks for each pose
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #convert img h w c ratio to x and y coordinates
                cx, cy = int(lm.x*w), int(lm.y * h)
                lmList.append([id, cx, cy])

        return lmList

