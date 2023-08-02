import pygame
import time

def main():
    pygame.init()
    pygame.mixer.init()

    # Load and play the music file
    pygame.mixer.music.load('sound/bluestone-alley.mp3')
    pygame.mixer.music.play(-1)  # -1 means play the music on loop

    # Set the initial volume to 0.5 (50% volume)
    volume = 1.0
    pygame.mixer.music.set_volume(volume)

    while True:
        time.sleep(5)
        volume -= 0.1
        print(f"Current Volume : {volume}")
        pygame.mixer.music.set_volume(volume)

if __name__ == "__main__":
    main()