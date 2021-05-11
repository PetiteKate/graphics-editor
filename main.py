import sys
import cv2
import numpy as np
import qimage2ndarray
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QPen
from PyQt5.QtWidgets import QGraphicsScene, QMainWindow, QMessageBox, QFileDialog, QInputDialog, QApplication
import MainWindow


class MyScene(QGraphicsScene):
    mousePressed = pyqtSignal(QPointF)
    mouseMoved = pyqtSignal(QPointF)

    def __init__(self):
        super().__init__()

    def mousePressEvent(self, event):
        point = event.scenePos()
        self.mousePressed.emit(point)

    def mouseMoveEvent(self, event):
        point = event.scenePos()
        self.mouseMoved.emit(point)


class Window(QMainWindow, MainWindow.Ui_MainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setupUi(self)

        self._image = None
        self._image_selected = False
        self._roi_active = False
        self._roi_first_point = None

        self.scene = MyScene()
        self.graphicsView.setScene(self.scene)
        self.scene.mousePressed.connect(self.click_on_picture)
        self.scene.mouseMoved.connect(self.move_on_picture)
        self.graphicsView.setMouseTracking(True)

        self.actionLoad_image.triggered.connect(self.open_image)
        self.actionLoad_image_from_XML.triggered.connect(self.open_xml)
        self.actionSave.triggered.connect(self.save_image_as_xml)
        self.actionExit.triggered.connect(self.exit)

        self.actionResize.triggered.connect(self.resize_picture)
        self.actionRotate.triggered.connect(self.rotate)
        self.actionCrop.triggered.connect(self.crop_image)
        self.actionROI.triggered.connect(self.roi)

        self.actionSmooth.triggered.connect(self.blur_filter)
        self.actionLighter.triggered.connect(self.lighter_filter)
        self.actionDarker.triggered.connect(self.darker_filter)
        self.actionSobel_operator.triggered.connect(self.sobel_operator)

        self.actionCanny_Detector.triggered.connect(self.canny_detector)
        self.actionHough_converting.triggered.connect(self.converting_hough)
        self.actionObject_Detection.triggered.connect(self.detect_object)
        self.actionComparison_of_contours.triggered.connect(self.detect_object_by_moment)

        self.actionHelp.triggered.connect(self.help)
        self.actionAbout.triggered.connect(self.about)

        self.disable_all_actions()

    def disable_all_actions(self):
        self.actionResize.setEnabled(False)
        self.actionRotate.setEnabled(False)
        self.actionCrop.setEnabled(False)
        self.actionROI.setEnabled(False)

        self.actionSmooth.setEnabled(False)
        self.actionLighter.setEnabled(False)
        self.actionDarker.setEnabled(False)
        self.actionSobel_operator.setEnabled(False)

        self.actionCanny_Detector.setEnabled(False)
        self.actionHough_converting.setEnabled(False)
        self.actionObject_Detection.setEnabled(False)
        self.actionComparison_of_contours.setEnabled(False)

    def enable_all_actions(self):
        self.actionResize.setEnabled(True)
        self.actionRotate.setEnabled(True)
        self.actionCrop.setEnabled(True)
        self.actionROI.setEnabled(True)

        self.actionSmooth.setEnabled(True)
        self.actionLighter.setEnabled(True)
        self.actionDarker.setEnabled(True)
        self.actionSobel_operator.setEnabled(True)

        self.actionCanny_Detector.setEnabled(True)
        self.actionHough_converting.setEnabled(True)
        self.actionObject_Detection.setEnabled(True)
        self.actionComparison_of_contours.setEnabled(True)

    def show_info(self, message: str):
        QMessageBox.information(self, 'Information', message)

    def show_error(self, message: str):
        QMessageBox.information(self, 'Error', message)

    def draw_image(self):
        if not self._image_selected:
            self.show_info('File not selected')
            return

        self.scene.clear()
        image_rgb = np.zeros((self._image.shape[0], self._image.shape[1], 3), np.uint8)
        cv2.cvtColor(src=self._image, code=cv2.COLOR_BGR2RGB, dst=image_rgb)
        image_bin = qimage2ndarray.array2qimage(image_rgb)
        pixMap = QPixmap.fromImage(image_bin)
        self.scene.addPixmap(pixMap)
        self.graphicsView.fitInView(QRectF(0, 0, image_rgb.shape[1], image_rgb.shape[0]),
                                    QtCore.Qt.KeepAspectRatio)

        # Это нужно для ROI т.к. после scene.clear остается синий квадрат, после выделения
        items_list = self.scene.items()
        if len(items_list) > 1:
            self.scene.removeItem(items_list[0])

        self.scene.update()

    def open_image(self):
        result = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        path = result[0]

        if not path:
            return

        self._image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if self._image is None:
            self.show_error('Error reading')

        self._image_selected = True
        self.draw_image()
        self.enable_all_actions()

    def open_xml(self):
        result = QFileDialog.getOpenFileName(self, 'Open file', '', "XML files (*.xml)")
        path = result[0]

        if not path:
            return

        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        self._image = cv_file.getNode("my_matrix").mat()
        cv_file.release()

        self._image_selected = True
        self.draw_image()
        self.enable_all_actions()

    def save_image_as_xml(self):
        if not self._image_selected:
            self.show_info('File not selected')
            return

        result = QFileDialog.getSaveFileName(self, 'Save file', '', 'XML files (*.xml)')
        path = result[0]

        if not path:
            return

        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("my_matrix", self._image)
        cv_file.release()

    def exit(self):
        self.close()

    def resize_picture(self):
        x, ok = QInputDialog.getInt(self, 'Input Dialog', 'Enter X:')
        if ok:
            y, okk = QInputDialog.getInt(self, 'Input Dialog', 'Enter Y:')
            if okk:
                if x <= 0 or y <= 0:
                    self.show_error('Wrong values')
                    return

                dim = (y, x)
                resized = cv2.resize(self._image, dim)
                self._image = resized
                self.draw_image()

    def rotate(self):
        # Сделано через матрицу поворота т.к.встроенный поворот режет картинку

        angle, ok = QInputDialog.getDouble(self, 'Input Dialog', 'Enter angle:')
        if ok:
            (h, w) = self._image.shape[:2]
            (cX, cY) = (w // 2, h // 2)

            # grab the rotation matrix (applying the negative of the
            # angle to rotate clockwise), then grab the sine and cosine
            # (i.e., the rotation components of the matrix)
            M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY

            # perform the actual rotation and return the image
            rotated = cv2.warpAffine(self._image, M, (nW, nH))
            self._image = rotated
            self.draw_image()

    def crop_image(self):

        x1, ok = QInputDialog.getInt(self, 'Input dialog', 'x1: ', min=0)
        if not ok:
            return

        x2, ok = QInputDialog.getInt(self, 'Input dialog', 'x2: ', min=0)
        if not ok:
            return

        y1, ok = QInputDialog.getInt(self, 'Input dialog', 'y1: ', min=0)
        if not ok:
            return

        y2, ok = QInputDialog.getInt(self, 'Input dialog', 'y2: ', min=0)
        if not ok:
            return

        rows, cols, _ = self._image.shape

        bad_args = y1 < 0 or x1 < 0 or y2 > rows or x2 > cols or y1 >= y2 or x1 >= x2
        if bad_args:
            msg = QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText('Wrong input')
            msg.exec()
            return

        croped = self._image[y1:y2, x1:x2]
        self._image = croped
        self.draw_image()

    def roi(self):
        self._roi_active = True
        self._roi_first_point = None

    def move_on_picture(self, point):
        x = point.x()
        y = point.y()

        if not self._roi_active:
            return

        if self._roi_first_point is None:
            return

        x1, y1 = self._roi_first_point

        a = self.scene.items()
        if len(a) > 1:
            self.scene.removeItem(a[0])

        width = x - x1
        height = y - y1
        pen = QPen(QtCore.Qt.blue)
        r = QtCore.QRectF(x1, y1, width, height)
        self.scene.addRect(r, pen)
        self.scene.update()

    def click_on_picture(self, point):
        x = point.x()
        y = point.y()

        if not self._roi_active:
            return

        if self._roi_first_point is None:
            self._roi_first_point = (x, y)
            return

        x1, y1 = self._roi_first_point
        ok = True
        rows, cols, _ = self._image.shape

        bad_args = y1 < 0 or x1 < 0 or y > rows or x > cols or y1 >= y or x1 >= x
        if bad_args:
            ok = False

        self._image = self._image[int(y1):int(y), int(x1):int(x)]

        if not ok:
            self.show_error("Bad arguments")
            return

        self.draw_image()

        self._roi_active = False
        self._roi_first_point = None

    def blur_filter(self):

        blur = cv2.blur(self._image, (5, 5))
        self._image = blur
        self.draw_image()

    def lighter_filter(self):
        kernel_lighten = [[-0.1, 0.2, -0.1], [0.2, 3, 0.2], [-0.1, 0.2, -0.1]]
        kernel = np.zeros((3, 3))
        kernel[:] = kernel_lighten

        dst = np.zeros((self._image.shape[0], self._image.shape[1], 3), np.uint8)
        cv2.filter2D(src=self._image, dst=dst, ddepth=-1, kernel=kernel)
        self._image = dst
        self.draw_image()

    def darker_filter(self):
        kernel_darken = [[-0.1, 0.1, -0.1], [0.1, 0.5, 0.1], [-0.1, 0.1, -0.1]]
        kernel = np.zeros((3, 3))
        kernel[:] = kernel_darken

        dst = np.zeros((self._image.shape[0], self._image.shape[1], 3), np.uint8)
        cv2.filter2D(src=self._image, dst=dst, ddepth=-1, kernel=kernel)
        self._image = dst
        self.draw_image()

    def sobel_operator(self):
        img_16 = self._image * 256

        img_sobel_8 = self._image.copy()
        img_sobel_16 = img_16.copy()

        cv2.Sobel(src=img_16, ddepth=-1, dst=img_sobel_16, dx=1, dy=1, ksize=3)
        cv2.convertScaleAbs(src=img_sobel_16, dst=img_sobel_8)
        self._image = img_sobel_8

        self.draw_image()

    def canny_detector(self):
        ratio = 3
        kernel_size = 3
        low_threshold = 50

        src_gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.blur(src_gray, (3, 3))
        detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
        mask = detected_edges != 0
        self._image = self._image * (mask[:, :, None].astype(self._image.dtype))
        self.draw_image()

    def converting_hough(self):
        src_gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.blur(src_gray, (3, 3))

        img_curr_p = img_canny.copy()
        lines = cv2.HoughLinesP(img_curr_p, rho=1, theta=np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)
        img_curr_p = cv2.cvtColor(src=img_curr_p, code=cv2.COLOR_GRAY2BGR)

        if lines is None:
            img_curr_p = img_canny.copy()
        else:
            for l in lines:
                cv2.line(img_curr_p, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (0, 255, 255), 3, cv2.LINE_AA)

        self._image = img_curr_p
        self.draw_image()

    def detect_object(self):

        available_detect_methods = (
            'TM_CCOEFF',
            'TM_CCOEFF_NORMED',
            'TM_CCORR',
            'TM_CCORR_NORMED',
            'TM_SQDIFF',
            'TM_SQDIFF_NORMED',
        )
        _available_detect_methods = {
            'TM_CCOEFF': cv2.TM_CCOEFF,
            'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
            'TM_CCORR': cv2.TM_CCORR,
            'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
            'TM_SQDIFF': cv2.TM_SQDIFF,
            'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED,
        }

        result = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        path_to_pattern_image = result[0]

        if path_to_pattern_image == '':
            return

        methods = available_detect_methods
        input_method_raw, ok = QInputDialog.getItem(self, "Select detect method dialog", "Detect method", methods, 0, False)

        if not ok or not input_method_raw:
            return

        method = _available_detect_methods.get(input_method_raw)
        if method is None:
            self.show_error('Unavailable method')
            return

        template = cv2.imread(path_to_pattern_image, cv2.IMREAD_UNCHANGED)
        if template is None:
            self.show_error('Error reading template file')
            return

        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)

        shape = gray_template.shape[::-1]
        width, height = (shape[0], shape[1])

        res = cv2.matchTemplate(image_gray, gray_template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + width, top_left[1] + height)
        cv2.rectangle(self._image, top_left, bottom_right, 255, 2)
        self.draw_image()

    def detect_object_by_moment(self):

        result = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        path_to_contour_image = result[0]

        if path_to_contour_image == '':
            return

        template = cv2.imread(path_to_contour_image, cv2.IMREAD_UNCHANGED)
        if template is None:
            self.show_error('Template file error reading')
            return

        template_res = cv2.Canny(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), 10, 200)
        image_res = cv2.Canny(cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY), 10, 200)

        contours_template, _ = cv2.findContours(template_res, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        contours_image, _ = cv2.findContours(image_res, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        if len(contours_template) != 1:
            self.show_error('Not founded templates')
            return

        contour_what_to_find = contours_template[0]
        m_min = 1000
        ind = 0
        ind_max = 0

        for contour in contours_image:
            m = cv2.matchShapes(contour, contour_what_to_find, cv2.CONTOURS_MATCH_I3, 0)

            if m < m_min:
                m_min = m
                ind_max = ind

            ind = ind + 1

        cv2.drawContours(self._image, contours_image, contourIdx=ind_max, color=(0, 0, 200), thickness=cv2.FILLED)
        self.draw_image()

    def help(self):
        self.show_info('This program can edit photo, for example rotate, resize, use 4 filetrs, and 4 operators')

    def about(self):
        self.show_info('Pashkovskaya Ekaterina\n 8-T30-302b-16')


def main():
    app = QApplication(sys.argv)
    mainWindow = Window()
    mainWindow.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
