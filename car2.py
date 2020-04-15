import sys
import numpy as np

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2

def warpInv(p):
    inverse_output = np.matrix([[0.1]] * 6)
    val = (1 + p[0, 0]) * (1 + p[3, 0]) - p[1, 0] * p[2, 0]
    inverse_output[0, 0] = (-p[0, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
    inverse_output[1, 0] = (-p[1, 0]) / val
    inverse_output[2, 0] = (-p[2, 0]) / val
    inverse_output[3, 0] = (-p[3, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
    inverse_output[4, 0] = (-p[4, 0] - p[3, 0] * p[4, 0] + p[2, 0] * p[5, 0]) / val
    inverse_output[5, 0] = (-p[5, 0] - p[0, 0] * p[5, 0] + p[1, 0] * p[4, 0]) / val
    return inverse_output
def get_New_Coordinate(Original, frame, x, y, size, gradOriginalX, gradOriginalY):
    if (((y + size) > len(Original)) or ((x + size) > len(Original[0]))): return np.matrix([[-1], [-1]])
    T = np.matrix([[Original[i, j] for j in range(x, x + size)] for i in range(y, y + size)])
    x1 = np.matrix([[q for q in range(size)] for z in range(size)])
    y1 = np.matrix([[z] * size for z in range(size)])

    gradOriginalX = np.matrix([[gradOriginalX[i, j] for j in range(x, x + size)] for i in range(y, y + size)])
    gradOriginalY = np.matrix([[gradOriginalY[i, j] for j in range(x, x + size)] for i in range(y, y + size)])

    gradOriginalP = [np.multiply(x1, gradOriginalX), np.multiply(x1, gradOriginalY), np.multiply(y1, gradOriginalX),np.multiply(y1, gradOriginalY), gradOriginalX, gradOriginalY]

    HessianOriginal = [[np.sum(np.multiply(gradOriginalP[a], gradOriginalP[b])) for a in range(6)] for b in range(6)]
    Hessianinv = np.linalg.pinv(HessianOriginal)

    p1, p2, p3, p4, p5, p6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    k = 0
    bad_itr = 0
    min_cost = -1
    minW = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    W = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    while (k <= 10):
        position = [[W.dot(np.matrix([[x + i], [y + j], [1]], dtype='float')) for i in range(size)] for j in range(size)]
        if not (0 <= (position[0][0])[0, 0] < cols and 0 <= (position[0][0])[1, 0] < rows and 0 <= position[size - 1][0][
            0, 0] < cols and 0 <= position[size - 1][0][1, 0] < rows and 0 <= position[0][size - 1][0, 0] < cols and 0 <=
            position[0][size - 1][1, 0] < rows and 0 <= position[size - 1][size - 1][0, 0] < cols and 0 <=
            position[size - 1][size - 1][1, 0] < rows):
            return np.matrix([[-1], [-1]])

        I = np.matrix([[frame[int((position[i][j])[1, 0]), int((position[i][j])[0, 0])] for j in range(size)] for i in range(size)])

        error = np.absolute(np.matrix(I, dtype='int') - np.matrix(T, dtype='int'))

        steepest_error = np.matrix([[np.sum(np.multiply(g, error))] for g in gradOriginalP])
        mean_cost = np.sum(np.absolute(steepest_error))
        deltap = Hessianinv.dot(steepest_error)
        dp = warpInv(deltap)
        p1, p2, p3, p4, p5, p6 = p1 + dp[0, 0] + p1 * dp[0, 0] + p3 * dp[1, 0], p2 + dp[1, 0] + dp[0, 0] * p2 + p4 * dp[1, 0], p3 + dp[2, 0] + p1 * dp[2, 0] + p3 * dp[3, 0], p4 + dp[3, 0] + p2 * dp[2, 0] + p4 * dp[3, 0], p5 + \
                                 dp[4, 0] + p1 * dp[4, 0] + p3 * dp[5, 0], p6 + dp[5, 0] + p2 * dp[4, 0] + p4 * dp[5, 0]
        W = np.matrix([[1+p1,p3,p5], [p2,1+p4,p6]])

        if (min_cost == -1):
            min_cost = mean_cost
        elif (min_cost >= mean_cost):
            min_cost = mean_cost
            bad_itr = 0
            minW = W
        else:
            bad_itr += 1
        if (bad_itr == 3):
            W = minW
            return W.dot(np.matrix([[x], [y], [1.0]]))

        if (np.sum(np.absolute(deltap)) < 0.0006):
            return W.dot(np.matrix([[x], [y], [1.0]]))


cap = cv2.VideoCapture("../Lucas_Kanade_Template_Tracker/car.avi")
# cap = cv2.VideoCapture("slow_traffic_small.mp4")
# feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
rows, cols = len(old_gray), len(old_gray[0])

# rows, cols = len(old_gray), len(old_gray[0])
# p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# feature_point = [p.ravel() for p in p0]
# feature_point = feature_point[:1]
#mask = np.zeros_like(old_frame)
def select_point(event, x, y, flags, params):
    global point, point1, point2, point_selected, point_selected_1, old_points_1, old_points_2, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        point = [(x, y)]
        point_selected = True
        # old_points_1 = np.array([[x, y]], dtype=np.float32)

    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        point.append((x, y))
        old_points = np.array(point, dtype=np.float32)
        point_selected_1 = True


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)
point_selected = False
point_selected_1 = False
point1 = ()
point2 = ()
point = []
old_points = np.array([[]])

while (len(old_points) > 0):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if point_selected and point_selected_1 is True:
        cv2.rectangle(frame, point1, point2, (0, 255, 0), 2)


        gradOriginalX = cv2.Sobel(old_gray, cv2.CV_32F, 1, 0, ksize=5)
        gradOriginalY = cv2.Sobel(old_gray, cv2.CV_32F, 0, 1, ksize=5)
        #good_new = [get_New_Coordinate(old_gray, frame_gray, int(x), int(y), 15, gradOriginalX, gradOriginalY) for x,
        # y in feature_point]
        good_new = [get_New_Coordinate(old_gray, frame_gray, int(x), int(y), 15, gradOriginalX, gradOriginalY) for x, y in old_points]

        # newfeature_point = []
        # # draw the tracks
        # for i in range(len(feature_point)):
        #     a, b = feature_point[i]
        #     c, d = int((good_new[i])[0]), int((good_new[i])[1])
        #     if (0 <= c < cols and 0 <= d < rows):
        #         mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        #         frame = cv2.circle(frame,(a,b), 5, color[i].tolist(), -1)
        #         newfeature_point.append((c,d))
        # img = cv2.add(frame,mask)
        old_gray = frame_gray.copy()
        # feature_point = newfeature_point[:]
        old_points = good_new

        good_new = np.array(good_new)

        x1, y1, x2, y2 = good_new.ravel()
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()