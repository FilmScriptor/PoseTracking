import cv2
import time
import PoseTrackingModule as pm


def main():
    # open video file
    cap = cv2.VideoCapture(0)

    # previous time
    pTime = 0

    detector = pm.poseDectector()

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        succes, img = cap.read()
        img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[10])

        # calculate frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # put fps text on screen
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()