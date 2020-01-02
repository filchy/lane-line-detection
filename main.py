from line_detection import process_frame

import cv2

def main():
	cap = cv2.VideoCapture("test_video1.mp4")
	while cap.isOpened():
		ret, color_frame = cap.read()
		line_frame = process_frame(color_frame)
		cv2.imshow("main", line_frame)
		if cv2.waitKey(35) == 27:
			break
	cv2.destroyAllWindows()
	cap.release()


if __name__ == '__main__':
	main()
