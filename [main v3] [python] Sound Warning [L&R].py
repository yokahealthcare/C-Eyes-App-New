import pygame
import time

# Initialize Pygame
pygame.init()

# Get a free channel
channel = pygame.mixer.find_channel()

# Set the volume for left and right speakers
left_volume = 0.0
right_volume = 0.0
# Set the volume for left and right speakers
channel.set_volume(left_volume, right_volume)

# Load and play the audio file
sound = pygame.mixer.Sound('sound/bluestone-alley.mp3')
channel.play(sound)


while True:
	left_volume += 0.1
	time.sleep(1)
	print(f"left volume : {left_volume}")
	channel.set_volume(left_volume, right_volume)


# Clean up
pygame.quit()
