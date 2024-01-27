import cv2
def count_people(video_path):
    cap = cv2.VideoCapture(video_path)
    full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    people_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            people_count += 1
        cv2.imshow('People Counting', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print('Total People Count: {people_count}')


if __name__ == "__main__":
    video_path = 'random vid.mp4'
    count_people(video_path)
