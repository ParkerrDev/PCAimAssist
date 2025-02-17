# Portion of screen to be captured (This forms a square/rectangle around the center of screen)
screenShotHeight = 640  # Reduced for better performance
screenShotWidth = 640  # Reduced for better performance

# Use "left" or "right" for the mask side depending on where the interfering object is, useful for 3rd player models or large guns
useMask = False
maskSide = "left"
maskWidth = 300
maskHeight = 375

# Autoaim mouse movement amplifier - reduced for more precise control
aaMovementAmp = 0.7

# Increased confidence threshold for more reliable detections
confidence = 0.45

# What key to press to quit and shutdown the autoaim
aaQuitKey = "P"

# If you want to main slightly upwards towards the head
headshot_mode = True

# Displays the Corrections per second in the terminal
cpsDisplay = True

# Set to True if you want to get the visuals
visuals = True

# Smarter selection of people
centerOfScreen = True

# ONNX ONLY - Choose 1 of the 3 below
# 1 - CPU
# 2 - AMD
# 3 - NVIDIA
onnxChoice = 3

# Set to True for third-person games, False for first-person games
ThirdPerson = True

# OBS Websocket Host
obs_host = "10.0.0.251"

# OBS Websocket Port
obs_port = 4455

# Offset to fine-tune aim (positive values move aim right/down)
xOffset = -175
yOffset = -185