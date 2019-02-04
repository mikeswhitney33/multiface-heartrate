from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import numpy as np

import face_recognition
import cv2

import evm

def faces_abstraction(video_capture, with_kivy=True, self=None):

    # Initialize some variables
    face_locations = []

    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)

        if face_locations:
            render = True
        else:
            render = False

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
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, str(np.random.randint(65,72)), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            faces.append(frame[top:bottom, left:right,:])

        if with_kivy:
            if render:
                self.write_frame(faces)
            elif not render:
                self.write_frame([frame])
                render = not render
        else:
            # Display the resulting image
            for i, frame in enumerate(faces):
                cv2.imshow('Video{}'.format(i), frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

def just_cv2():
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    faces_abstraction(video_capture, with_kivy=False)

class CamApp(App):

    def build(self):
        self.img1=Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)

        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def write_frame(self, frames):
        max_0 = -float("inf")
        max_1 = -float("inf")
        all_shape_0 = 0
        all_shape_1 = 1

        print(frames[0].shape)

        buffer = bytearray()
        for frame in frames:
            all_shape_0 += frame.shape[0]
            all_shape_1 += frame.shape[1]
            if frame.shape[0] > max_0:
                max_0 = frame.shape[0]
            if frame.shape[1] > max_1:
                max_1 = frame.shape[1]


            buffer.extend(bytearray(cv2.flip(frame, 0).tostring()))

        texture1 = Texture.create(size=(max_1, all_shape_0), colorfmt='bgr')
        texture1.blit_buffer(bytes(buffer), colorfmt='bgr', bufferfmt='ubyte')
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

        if face_locations:
            render = True
        else:
            render = False

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
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, str(np.random.randint(65,72)), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            faces.append(frame[top:bottom, left:right,:])


        if render:
            self.write_frame(faces)
        elif not render:
            self.write_frame([frame])
            render = not render

if __name__ == '__main__':
    just_cv2()
    # CamApp().run()
