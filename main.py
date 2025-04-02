import mediapipe as mp #Open source framework,used for media processing
import cv2 #open source lib used to perform tasks like face detection,object tracking.
import numpy as np #python lib used for working with arrays
import time #allows to work with time in python

#constants
ml = 150# Margin from the left for tool selection
max_x, max_y = 250+ml, 50 # Toolbar dimensions
curr_tool = "select tool"
time_init = True# Flag to track tool selection time
rad = 40# Radius for tool selection effect
var_inits = False# Flag for drawing shapes
thick = 4# Thickness of drawing
prevx, prevy = 0,0# Previous coordinates for drawing

#get tools function
def getTool(x):
	if x < 50 + ml:
		return "line"

	elif x<100 + ml:
		return "rectangle"

	elif x < 150 + ml:
		return"draw"

	elif x<200 + ml:
		return "circle"

	else:
		return "erase"

def index_raised(yi, y9):#check if index finger is raised or not
	if (y9 - yi) > 40:#index finger base -index finger tip
		#y9 is middle finger base
		return True

	return False
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)#how accurate is the bounding box.
draw = mp.solutions.drawing_utils#hand detection


# drawing tools
tools = cv2.imread('images/tools.png') #imread() method loads an image from the specified file
tools = tools.astype('uint8') # Convert to unsigned 8-bit format.

mask = np.ones((480, 640))*255#creates a 2d array with white canvas stores the drawing of camera feed ..480 rows and 640 col 255 becoz we need a white canvas
#When you draw something (line, rectangle, etc.), it modifies the same mask.
# The mask does not reset every frame, so drawings stay visible.
# If you close and restart the program, the mask is reinitialized as a new blank white canvas.
mask = mask.astype('uint8')
'''
tools = np.zeros((max_y+5, max_x+5, 3), dtype="uint8")
cv2.rectangle(tools, (0,0), (max_x, max_y), (0,0,255), 2)
cv2.line(tools, (50,0), (50,50), (0,0,255), 2)
cv2.line(tools, (100,0), (100,50), (0,0,255), 2)
cv2.line(tools, (150,0), (150,50), (0,0,255), 2)
cv2.line(tools, (200,0), (200,50), (0,0,255), 2)
'''
cap = cv2.VideoCapture(0)#opens default webcam
while True:
	_, frm = cap.read()#read the frames using the created objects
	frm = cv2.flip(frm, 1)# Mirror the image for a natural user experience. flips image horizontally

	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)#convert from 1 colour space to another

	op = hand_landmark.process(rgb)# Detect hands in the frame.

	if op.multi_hand_landmarks:#detect hand
		for i in op.multi_hand_landmarks:
			draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)#draw hand landmarks
			x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)#get index finger position Converts the position to pixel values.

			if x < max_x and y < max_y and x > ml:#check if the indexfinger is within the tool area

				#if index finger stays in this area for more than 0.8 sec then the tool is selected
				if time_init:
					ctime = time.time()#start timer
					time_init = False
				ptime = time.time()

				cv2.circle(frm, (x, y), rad, (0,255,255), 2)#shows yellow circle on fingertip
				rad -= 1

				if (ptime - ctime) > 0.8:
					curr_tool = getTool(x)
					print("your current tool set to : ", curr_tool)
					time_init = True
					rad = 40

			else:
				time_init = True
				rad = 40
#check if middle finger is raised if yes then we can draw then calculate dist from start to end of current pos
			if curr_tool == "draw":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
					prevx, prevy = x, y

				else:#if finger is not raised reset the prev pos
					prevx = x
					prevy = y


			elif curr_tool == "line":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)#middle finger tip
				y9  = int(i.landmark[9].y*480)#middle finger base

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y#stores the first point of line
						var_inits = True

					cv2.line(frm, (xii, yii), (x, y), (50,152,255), thick)#line is drawn from first point to current using color blue

				else:
					if var_inits:
						cv2.line(mask, (xii, yii), (x, y), 0, thick)#permanent line is drawn on mask using black color
						var_inits = False

			elif curr_tool == "rectangle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.rectangle(frm, (xii, yii), (x, y), (0,255,255), thick)

				else:
					if var_inits:
						cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif curr_tool == "circle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y#circle centre
						var_inits = True
#radius is calculated by distance formula between 2 points
					cv2.circle(frm, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick)

				else:
					if var_inits:#if finger not raised then permanent green circle is drawn on mask
						cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), thick)
						var_inits = False

			elif curr_tool == "erase":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):#draw circle as eraser
					cv2.circle(frm, (x, y), 30, (0,0,0), -1)
					cv2.circle(mask, (x, y), 30, 255, -1)



	op = cv2.bitwise_and(frm, frm, mask=mask)
	frm[:, :, 1] = op[:, :, 1]
	frm[:, :, 2] = op[:, :, 2]

	frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

	cv2.putText(frm, curr_tool, (270+ml,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("paint app", frm)#display the drawing
	#displays the video in real time

	if cv2.waitKey(1) == ord('q'):#press q to exit from application
		cv2.destroyAllWindows()
		cap.release()
		break