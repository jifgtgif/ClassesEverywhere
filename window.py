import pygame
import pygame

# Initialize Pygame
pygame.init()

# Set up the display
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("My Pygame Window")

# Colors
white = (255, 255, 255)
red = (255, 0, 0)

# Game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Drawing
    screen.fill(white)  # Fill the background with white
    pygame.draw.circle(screen, red, (screen_width // 2, screen_height // 2), 50) # Draw a red circle in the center

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()