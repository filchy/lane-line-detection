from line import Line

import numpy as np
import cv2
import math

img_h = 540 
img_w = 970
# percent value of polygon height -> 59%
polyg_height = 0.59 - 0.2 # 0.2 values is just for better looking lines indicators

vertices = np.array([[(50,img_h),
			(450,320),
			(510,320),
			(img_w-50,img_h)]], dtype=np.int32)


def region_of_interest(img, vertices):
	mask = np.zeros_like(img)

	if len(img.shape) > 2:
		channel_count = img.shape[2]
		ignore_mask_color = (255, ) * channel_count
	
	else:
		ignore_mask_color = 255

	cv2.fillPoly(mask, vertices, ignore_mask_color)

	masked_img = cv2.bitwise_and(img, mask)
	
	return masked_img, mask


def hough_lines_detection(img, rho, theta, min_line_lenght, max_line_gap):
	lines = cv2.HoughLinesP(img, rho, theta, min_line_lenght, max_line_gap)

	return lines


def compute_lane_from_candidates(line_candidates, img_shape):
	#line_candidates: array from hough_lines_detection

	# sorting lines by +(right lines) / -(left lines)
	pos_lines = [line for line in line_candidates if line.slope > 0 if math.isnan(line.slope) == False]
	neg_lines = [line for line in line_candidates if line.slope < 0 if math.isnan(line.slope) == False]

	# right line
	right_lines_x = []
	right_lines_y = []

	for line in pos_lines:
		x1, y1, x2, y2 = line.get_coords()
		right_lines_x.append(x1)
		right_lines_x.append(x2)
		right_lines_y.append(y1)
		right_lines_y.append(y2)

	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)

	else:
		right_m, right_b = 1, 1
	
	# left line
	left_lines_x = []
	left_lines_y = []

	for line in neg_lines:
		x1, y1, x2, y2 = line.get_coords()
		left_lines_x.append(x1)
		left_lines_x.append(x2)
		left_lines_y.append(y1)
		left_lines_y.append(y2)

	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)

	else:
		left_m, left_b = 1, 1

	y1 = np.int32(img_shape[0])
	y2 = np.int32(img_shape[0] * (1 - polyg_height))

	right_x1 = np.int32((y1 - right_b) / right_m)
	right_x2 = np.int32((y2 - right_b) / right_m)

	left_x1 = np.int32((y1 - left_b) / left_m)
	left_x2 = np.int32((y2 - left_b) / left_m)

	left_line = Line(left_x1, y1, left_x2, y2)
	right_line = Line(right_x1, y1, right_x2, y2)

	"""
	# right line
	pos_bias = np.median([line.bias for line in pos_lines if math.isnan(line.bias) == False]).astype(int)
	pos_slope = np.median([line.slope for line in pos_lines if math.isnan(line.slope) == False])
	x1, y1 = np.int32(np.round((img_shape[0] - pos_bias) / pos_slope)), img_shape[0]
	x2, y2 = x1 - 250, 350
	x2, y2 = np.int32(np.round(pos_bias / pos_slope)) + 600, 350
	right_line = Line(x1, y1, x2, y2)
	
	# left line
	neg_bias = np.median([line.bias for line in neg_lines if math.isnan(line.bias) == False]).astype(int)
	neg_slope = np.median([line.slope for line in neg_lines if math.isnan(line.slope) == False])
	x1,	 y1 = np.int32(np.round((img_shape[0] - neg_bias) / neg_slope)), img_shape[0]
	x2, y2 = -np.int32(np.round(neg_bias / neg_slope)) - 320, 350
	left_line = Line(x1, y1, x2, y2)
	"""	

	return left_line, right_line


def get_lane_lines(img, all_lines, solid_lines):
	detected_lines = [Line(l[0][0],l[0][1],l[0][2],l[0][3]) for l in all_lines]

	if solid_lines == True:
		candidate_lines = []

		for line in detected_lines:
			if 0.1 <= np.abs(line.slope) <= 3:
				candidate_lines.append(line)

		lane_lines = compute_lane_from_candidates(candidate_lines, img.shape)
	else:
		lane_lines = detected_lines

	return lane_lines


def process_frame(img, solid_lines=True):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img_blur = cv2.GaussianBlur(img_gray, (15,15), 0)

	img_canny = cv2.Canny(img_blur, 50, 80)

	img_h, img_w = img_canny.shape[0], img_canny.shape[1]
	"""
	vertices = np.array([[(50,img_h),
						(450,320),
						(510,320),
						(img_w-50,img_h)]],dtype=np.int32)
	"""
	img_masked, _ = region_of_interest(img_canny, vertices)

	all_lines = hough_lines_detection(img_masked, 2, np.pi/180, 10, 15)

	detected_lines = get_lane_lines(img_masked, all_lines, solid_lines)
	
	if detected_lines is not None:
		for line in detected_lines:
			line.draw(img)
	
	return img
