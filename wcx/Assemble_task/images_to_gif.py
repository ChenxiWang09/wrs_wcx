from PIL import Image, ImageDraw, ImageFont
import sys
# List the file paths of the images in the desired order
routine_filenames = []
for i in range(0, 9):
    routine_filename = "/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/routine_pictures/reality/t2/routine_" + str(
        i) + ".png"
    routine_filenames.append(routine_filename)
# Create a list to store the image frames
frames = []

# Load each image and append it to the frames list
for file_path in routine_filenames:
    image = Image.open(file_path)
    frames.append(image)

# Add a pause at the final frame by duplicating the last frame and increasing its duration
pause_duration = 1000  # Pause duration in milliseconds
frames.append(frames[-1].copy())
frames.append(frames[-1].copy())
frames.append(frames[-1].copy())
frame_durations = [pause_duration] * len(frames)  # Set the same duration for each frame
frame_durations.append(pause_duration)

# Save the frames as an animated GIF with custom durations
frames[0].save(
    "/home/hlabbunri/Desktop/chenxi wang/wrs-wcx/wcx/Assemble_task/data/routine_pictures/routine_6.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=frame_durations,
    loop=0
)
