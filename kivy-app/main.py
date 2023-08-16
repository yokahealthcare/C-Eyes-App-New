from kivy.core.image import Texture
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.clock import Clock
import ceyes as eye
import cv2

class MainApp(MDApp):
    def build(self):
        self.left_volume = 0.0
        self.right_volume = 0.0

        layout = MDBoxLayout(orientation="vertical")
        self.image = Image()

        layout.add_widget(self.image)
        layout.add_widget(MDRaisedButton(
            text="CLICK HERE",
            pos_hint={"center_x":.5, 'center_y':.5},
            size_hint=(None, None)
        ))

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)

        return layout

    def load_video(self, *agrs):
        ret, frame = self.capture.read()

        # process video
        # setting WIDTH variable with current screen size
        eye.WIDTH = frame.shape[1]
        frame = self.process_video(frame)

        # plot opencv to kivy widget
        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buffer, colorfmt="bgr", bufferfmt="ubyte")
        self.image.texture = texture

    def process_video(self, frame):
        faces = eye.detect_faces(frame)
        distance = eye.get_face_distance_approx(eye.focal_length, eye.known_width, faces)



        for x, y, w, h in faces:
            # draw the rectangle on the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), eye.GREEN, 2)

        return frame

if __name__ == "__main__":
    MainApp().run()