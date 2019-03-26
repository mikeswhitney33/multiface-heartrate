from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import numpy as np

import face_recognition
import cv2 as cv
import time

from evm import evm


"""
Some Constants for using within the app
"""
RED = (0, 0, 255)
WHITE = (255, 255, 255)
FONT = cv.FONT_HERSHEY_DUPLEX
RESCALE = 4


class FaceQueue(object):
    """
    FaceQueue:
    ----------
    A class used to keep track of an individual face
    """
    def __init__(self, encoding, id=0):
        """
        method __init__:
        ---------------
        initializes a FaceQueue

        Parameters:
        @encoding: the encoding of the face we will use to compare with other faces.
        @id: a unique identifier, mostly used for debuging purposes.
        """
        self.encoding = encoding
        self.queue = []
        self.times = []
        self.bpms = [0]
        self.t0 = time.time()
        self.id = id

    def add(self, center, fps):
        """
        method add:
        ----------
        adds a frame to its queue and calculates heart rate

        Parameters:
        @center: the frame to add to its queue
        @fps: the frames per second for calculating the heart rate
        """
        self.queue.append(center)
        self.times.append(time.time() - self.t0)

        if len(self.queue) > 10:
            self.queue = self.queue[-100:]
            self.times = self.times[-100:]
            self.bpms = self.bpms[-100:]
            frames = np.array(self.queue)

            try:
                hr = evm.find_heart_rate(frames, self.times, fps, .6, 1, alpha=20)

                self.bpms.append(hr)
            except:
                pass

    def get_hr(self):
        """
        method get_hr:
        -------------
        retrieves the heart rate from the queue

        Returns:
        the heart rate calculated from the queue
        """
        return np.mean(self.bpms)


class CamApp(App):

    def build(self):
        """
        method build:
        ------------
        sets up the UI of the App

        Returns:
        The root widget
        """
        layout = BoxLayout()
        self.image = Image()
        layout.add_widget(self.image)

        self.capture = cv.VideoCapture(0)

        self.queues = []

        Clock.schedule_interval(self.update, 1/33)
        return layout

    def get_queue(self, encoding, tolerence=0.6):
        """
        method get_queue:
        ----------------

        Parameters:
        @encoding: the encoding of the face we are looking for
        @tolerence: how similar two faces have to be to be considered the same.
            Defaults to .6

        Returns:
        the index into the queue list if the face exists, -1 otherwise
        """
        for i, q in enumerate(self.queues):
            diff = np.linalg.norm(encoding - q.encoding)
            if diff < tolerence:
                return i
        return -1

    def write_frame(self, frame):
        """
        method write_frame:
        ------------------
        writes an image to the screen

        Parameters:
        @frame: the frame we will be rendering
        """
        h, w, c = frame.shape
        buf = frame[::-1].tostring()

        texture = Texture.create(size=(w, h), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def draw_heart_rate(self, frame, hr, top, right, bottom, left, id):
        """
        method draw_heart_rate:
        ----------------------
        draws the bounding box and heart rate on a frame

        Parameters:
        @frame: the frame for which we will do the drawing (is modified inplace)
        @hr: the heart rate
        @top, right, bottom, left: the extents of the bounding box
        @id: an identifier for which face we are looking at, we could remove
            this for final product
        """
        cv.rectangle(frame, (left, top), (right, bottom), RED, 2)
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), RED, cv.FILLED)
        txt = "{}: {:.0f}".format(id, hr)
        cv.putText(frame, txt, (left + 6, bottom - 6), FONT, 1, WHITE, 1)

    def crop_center(self, im, size):
        """
        method crop_center:
        ------------------
        performs a center crop on an image

        Parameters:
        @im: the input image
        @size: either a tuple (width, height) or int for uniform size

        Returns:
        The cropped image
        """
        y, x, _ = im.shape
        if hasattr(size, '__len__'):
            cropx, cropy = size
        else:
            cropx = size
            cropy = size
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return im[starty:starty+cropy,startx:startx+cropx]

    def update(self, dt):
        """
        method update:
        -------------
        updates the app, this is called every 1/30 seconds

        Parameters:
        @dt: the elapsed time between this update and the last
        """
        # Read in frame
        ret, frame = self.capture.read()

        # perform face detection on smaller frame for performance
        small_frame = cv.resize(frame, (0, 0), fx=1/RESCALE, fy=1/RESCALE)
        face_locations = face_recognition.face_locations(small_frame[:,:,::-1])

        # get the face encodings to compare between faces
        encodings = face_recognition.face_encodings(small_frame, face_locations)

        for encoding, loc in zip(encodings, face_locations):
            # rescale the bounding boxes
            top, right, bottom, left = (a*RESCALE for a in loc)
            # get the face from the frame
            rect = frame[top:bottom,left:right]
            # get the very center for more accurate heart rate
            center = self.crop_center(rect, 80)

            fps = self.capture.get(cv.CAP_PROP_FPS)
            # update the face queue
            qi = self.get_queue(encoding)
            if qi == -1:
                q = FaceQueue(encoding, len(self.queues))
                self.queues.append(q)
            self.queues[qi].add(center, fps)

            hr = self.queues[qi].get_hr()
            # draw the bounding box and heart rate on the frame
            self.draw_heart_rate(frame, hr, top, right, bottom, left, self.queues[qi].id)
        # display the frame
        self.write_frame(frame)


if __name__ == '__main__':
    CamApp().run()
