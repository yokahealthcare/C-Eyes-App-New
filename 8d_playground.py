import pygame
import time


def main():
    # initialize pygame
    pygame.init()
    pygame.mixer.init()

    # Get a free channel
    channel = pygame.mixer.find_channel()

    # Set the volume for left and right speakers
    left_volume = 0.8
    right_volume = 0.2
    # Set the volume for left and right speakers
    channel.set_volume(left_volume, right_volume)

    # Load and play the audio file
    sound = pygame.mixer.Sound('sound/object-ding.mp3')
    channel.play(sound, -1)

    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    print("Spacebar pressed")
                elif event.key == pygame.K_UP:
                    print("Up arrow key pressed")


if __name__ == "__main__":
    main()
