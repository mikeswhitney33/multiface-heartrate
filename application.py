from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import numpy as np

import face_recognition
import cv2

from evm import evm

class CamApp(App):

    def build(self):
        self.img1=Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        self.queue = np.array([])

        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def write_frame(self, frame):
        buffer = cv2.flip(frame, 0).tostring()

        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1

    def update(self, dt):

        # Initialize some variables
        face_locations = []

        process_this_frame = True
        # Grab a single frame of video
        ret, frame = self.capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)

        process_this_frame = not process_this_frame

        # Display the results
        faces = []
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            
            # faces.append(frame[top:bottom, left:right,:])

            self.queue = np.append(self.queue, frame[top:bottom, left:right,:])

            heart_rate = 0
            if len(self.queue) >= 30:
                frames = np.array(self.queue[:30])
                self.queue = np.array([])
                heart_rate = evm.find_heart_rate(frames, 30, 0.8, 1.0, alpha=100)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, str(heart_rate), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        self.write_frame(frame)

if __name__ == '__main__':
    # just_cv2()
    CamApp().run()
    video_capture.release()
    cv2.destroyAllWindows()
