import multiprocessing as mp
import cv2
import numpy as np
import os

"""
2018-06-03 Yonv1943
2018-06-08 stable, compare the real time frames, and draw the contours of the moving objects
2018-07-02 setattr(), if is_opened
2018-11-24 polygon 
"""


class DrawROI(object):  # draw Region of Interest
    def __init__(self, img):
        self.img = img
        self.window_name = self.__class__.__name__

        self.done = False
        self.cur_pt = (0, 0)  # Current position, so we can draw the line-in-progress
        self.roi_pts = []  # List of points defining our polygon
        self.roi_pts_pwd = "%s_points.npy" % self.__class__.__name__

    def on_mouse(self, event, x, y, _buttons, _user_param):
        """
        drawing-filled-polygon-using-mouse-events-in-open-cv-using-python
        https://stackoverflow.com/questions/37099262/
        Original Code: Dan Ma拧ek
        Modified Code: Yonv1943
        """
        if self.done:  # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.cur_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.roi_pts), x, y))
            self.roi_pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.roi_pts))
            self.done = True

    def confirm(self, img):
        roi_mat = cv2.fillPoly(np.zeros(img.shape, dtype=np.uint8), self.roi_pts, (1, 1, 1))
        high_light_roi_mat = np.ones(roi_mat.shape, dtype=np.float) * 0.5
        high_light_roi_mat = cv2.fillPoly(high_light_roi_mat, self.roi_pts, (1.0, 1.0, 1.0))  # highlight the roi

        canvas = np.copy(img)
        canvas = canvas * high_light_roi_mat
        canvas = np.array(canvas, dtype=np.uint8)
        cv2.imshow(self.window_name, canvas)

        while os.path.exists(self.roi_pts_pwd):
            wait_key = cv2.waitKey(50)
            if wait_key == 13:
                self.done = True
                break
            elif wait_key == 8:  # ord('\b'), Backspace
                os.remove(self.roi_pts_pwd)  # delete the *.npy and redraw
                self.roi_pts = []  # initialize the pts
                self.done = False

    def draw_roi(self, line_color=(234, 234, 234)):
        """
        drawing-filled-polygon-using-mouse-events-in-open-cv-using-python
        https://stackoverflow.com/questions/37099262/
        Original Code: Dan Ma拧ek
        Modified Code: Yonv1943
        """
        img = self.img
        roi_mat = np.zeros(img.shape, dtype=np.uint8)  # Region of Image(ROI)

        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_FREERATIO)
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        canvas = np.copy(img)
        canvas_cur = canvas
        pre_pts_len = len(self.roi_pts)  # previous points of polygon number

        if os.path.exists(self.roi_pts_pwd):
            self.roi_pts = np.load(self.roi_pts_pwd)
            self.confirm(img)

        while not self.done:
            if len(self.roi_pts) != pre_pts_len:
                pre_pts_len = len(self.roi_pts)
                cv2.line(canvas_cur, self.roi_pts[-1], self.cur_pt, line_color, 2)
                canvas = canvas_cur
            else:
                canvas_cur = np.copy(canvas)

            if len(self.roi_pts) > 0:
                cv2.line(canvas_cur, self.roi_pts[-1], self.cur_pt, line_color, 2)

            cv2.imshow(self.window_name, canvas_cur)  # Update the window

            if cv2.waitKey(50) == 13:  # ENTER hit, ord('\r') == 13
                self.roi_pts = np.array([self.roi_pts])
                roi_mat = cv2.fillPoly(roi_mat, self.roi_pts, (1, 1, 1))
                print(self.roi_pts_pwd)
                np.save(self.roi_pts_pwd, self.roi_pts)

                self.confirm(img)

        cv2.destroyWindow(self.window_name)
        return self.roi_pts


class EdgeDetection(object):  # FrogEyes
    def __init__(self, img, roi_pts):
        self.min_thresh = 56.0
        self.roi_pts = roi_pts
        self.roi_mat = cv2.fillPoly(np.zeros(img.shape, dtype=np.uint8), self.roi_pts, (1, 1, 1))

        img = self.img_preprocessing(img)
        background_change_after_read_image_number = 64  # change time = 0.04*number = 2.7 = 0.4*64
        self.img_list = [img for _ in range(background_change_after_read_image_number)]
        self.high_light_roi_mat = cv2.fillPoly(np.ones(self.roi_mat.shape, dtype=np.float) * 0.25,
                                               self.roi_pts, (1.0, 1.0, 1.0), )  # highlight the roi

        self.img_len0 = int(360)
        self.img_len1 = int(self.img_len0 / (img.shape[0] / img.shape[1]))
        self.img_back = self.img_preprocessing(img)  # background

        self.min_side_num = 3
        self.min_side_len = int(self.img_len0 / 24)  # min side len of polygon
        self.min_poly_len = int(self.img_len0 / 12)
        self.thresh_blur = int(self.img_len0 / 8)

    def img_preprocessing(self, img):
        img = np.copy(img)
        img *= self.roi_mat
        # img = cv2.resize(img, (self.img_len1, self.img_len0))
        # img = cv2.bilateralFilter(img, d=3, sigmaColor=16, sigmaSpace=32)
        return img

    def get_polygon_contours(self, img, img_back):
        # img = np.copy(img)
        dif = np.array(img, dtype=np.int16)
        dif = np.abs(dif - img_back)
        dif = np.array(dif, dtype=np.uint8)  # get different

        gray = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, self.min_thresh, 255, 0)
        thresh = cv2.blur(thresh, (self.thresh_blur, self.thresh_blur))

        if np.max(thresh) == 0:  # have not different
            contours = []
        else:
            thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # hulls = [cv2.convexHull(cnt) for cnt, hie in zip(contours, hierarchy[0]) if hie[2] == -1]
            # hulls = [hull for hull in hulls if cv2.arcLength(hull, True) > self.min_hull_len]
            # contours = hulls

            approxs = [cv2.approxPolyDP(cnt, self.min_side_len, True) for cnt in contours]
            approxs = [approx for approx in approxs
                       if len(approx) > self.min_side_num and cv2.arcLength(approx, True) > self.min_poly_len]
            contours = approxs
        return contours

    def main_get_img_show(self, origin_img):
        img = self.img_preprocessing(origin_img)
        contours = self.get_polygon_contours(img, self.img_back)

        self.img_list.append(img)
        img_prev = self.img_list.pop(0)

        self.img_back = img \
            if not contours or not self.get_polygon_contours(img, img_prev) \
            else self.img_back

        show_img = np.array(self.high_light_roi_mat * origin_img, dtype=np.uint8)
        show_img = cv2.polylines(show_img, contours, True, (0, 0, 255), 2) if contours else show_img
        return show_img


def queue_img_put(q, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    while True:
        is_opened, frame = cap.read()
        q.put(frame) if is_opened else None
        q.get() if q.qsize() > 1 else None


def queue_img_get(q, window_name):
    frame = q.get()
    region_of_interest_pts = DrawROI(frame).draw_roi()

    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    frog_eye = EdgeDetection(frame, region_of_interest_pts)

    while True:
        frame = q.get()
        img_show = frog_eye.main_get_img_show(frame)
        cv2.imshow(window_name, img_show)
        cv2.waitKey(1)


def run():
    user_name, user_pwd, camera_ip = "admin", "password", "192.168.1.164"

    mp.set_start_method(method='spawn')  # multi-processing init
    queue = mp.Queue(maxsize=2)
    processes = [mp.Process(target=queue_img_put, args=(queue, user_name, user_pwd, camera_ip)),
                 mp.Process(target=queue_img_get, args=(queue, camera_ip))]

    [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    run()
pass
