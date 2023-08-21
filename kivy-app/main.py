from kivy.core.image import Texture
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.clock import Clock
import ceyes as eye
import cv2
import pygame

class MainApp(MDApp):
    def build(self):
        layout = MDBoxLayout(orientation="vertical")
        self.image = Image()

        layout.add_widget(self.image)
        # layout.add_widget(MDRaisedButton(
        #     text="CLICK HERE",
        #     pos_hint={"center_x":.5, 'center_y':.5},
        #     size_hint=(None, None)
        # ))

        # initialize pygame
        pygame.init()
        pygame.mixer.init()

        # Get a free channel
        self.channel = pygame.mixer.find_channel()

        # Set the volume for left and right speakers
        self.left_volume = 0.0
        self.right_volume = 0.0
        # Set the volume for left and right speakers
        self.channel.set_volume(self.left_volume, self.right_volume)

        # Load and play the audio file
        sound = pygame.mixer.Sound('../sound/alarm-fire.mp3')
        self.channel.play(sound, -1)
        
        
        # video capture started
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
        # detect faces
        faces = eye.detect_faces(frame)
        # calculate distance
        distance = eye.get_face_distance_approx(eye.focal_length, eye.known_width, faces)

        # normalize the distance range 0 to 1
        MIN = 30  # centimeter
        MAX = 100  # centimeter

        print(f"Distance : {distance}")

        if distance["L"] != []:
            left = (min(distance["L"]) - MIN) / (MAX - MIN)
            self.left_volume = 1.0 - left

        else:
            if self.left_volume > 0.0:
                # slowly decrease volume if no face detected
                self.left_volume -= 0.05

        if distance["R"] != []:
            right = (min(distance["R"]) - MIN) / (MAX - MIN)
            self.right_volume = 1.0 - right

        else:
            if self.right_volume > 0.0:
                # slowly decrease volume if no face detected
                self.right_volume -= 0.05

        # safety limit
        self.left_volume = max(0.0, min(self.left_volume, 1.0))
        self.right_volume = max(0.0, min(self.right_volume, 1.0))

        # Set the volume for left and right speakers
        self.channel.set_volume(self.left_volume, self.right_volume)

        for x, y, w, h in faces:
            # draw the rectangle on the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), eye.GREEN, 2)

        idx = 0
        for key, values in distance.items():
            for value in values:
                cv2.putText(frame, f"Person {idx+1}: {round(value,2)} CM", (30, 35+idx*30), eye.fonts, 0.6, eye.GREEN, 1)
                idx += 1

        return frame

if __name__ == "__main__":
    MainApp().run()