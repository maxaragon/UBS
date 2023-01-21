import sys
import glob
import time
import h5py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Raw data path
path = str(sys.argv[1])

# Select number of images to process
#batches = int(sys.argv[2])

start_time = time.time()

files = glob.glob(path + '/*.h5')

def get_data(file):
    '''
    get_data takes one .h5 SEVIR file as input and extract event_id and data
    '''
    if 'IR069' in file:
        with h5py.File(file,'r') as hf:
            event_id = hf['id'][:]
            data  = hf['ir069'][:]
    elif 'VIS' in file:
        with h5py.File(file,'r') as hf:
            event_id = hf['id'][:]
            data  = hf['vis'][:]
    elif 'VIL' in file:
        with h5py.File(file,'r') as hf:
            event_id = hf['id'][:]
            data  = hf['vil'][:]

    return event_id, data



def create_animation(ir, filename):
    #plt.ioff()
    # Create a figure and axes for the plot
    fig, ax = plt.subplots()

    # Create an empty plot
    im = ax.imshow(ir[:,:,0])

    # Define the function to update the plot
    def update(i):
        im.set_data(ir[:,:,i])
        return im,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(ir.shape[2]), repeat=True, blit=True)

    # Save the animation as a video file
    ani.save(filename, writer='pillow', fps=10)
    plt.close()


def kmeans(img):
    if len(img.shape) == 2:
        Z = img.reshape((-1,1))
    else:
        Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


# Get data from the first file
event_id, data = get_data(files[0])


events = data.shape[0]
image_count = data.shape[-1]

print(f'Number of files: {len(files)}, events/batches per file: {events}, images per event: {image_count}')

new_size = (112, 112)

# Create the new data array with the desired shape
new_data = np.empty((data.shape[0], new_size[0], new_size[1], data.shape[-1]))

# Loop through all the events and resize the images
for event in range(data.shape[0]):
    # Loop through all the images in the event
    for image in range(data.shape[-1]):
        # Get the current image
        current_image = data[event, :, :, image]
        # Resize the image
        current_image = cv.resize(current_image, new_size)
        # Save the resized image to the new data array
        new_data[event, :, :, image] = current_image

data = new_data 

# Perform the kmeans processing
clustered_images = np.zeros((events, image_count, data.shape[1], data.shape[2]))

for event in range(events): #events here to process all
    for image in range(image_count):
        clustered_img = kmeans(data[event, :, :, image])
        clustered_images[event, image, :, :] = clustered_img

clustered_images = np.swapaxes(clustered_images, 1, 3)

# Generate the animations
for event in range(events): #events here to process all
    filename_kmeans = f"event_kmeans_{event}.gif"
    filename_original = f"event_original_{event}.gif"
    create_animation(clustered_images[event, :, :, :], filename_kmeans)
    #print(f'processed animation {event}')
    create_animation(data[event, :, :, :], filename_original)

end_time = time.time()

# Calculate and print the time taken
time_taken = end_time - start_time
print(f'Time taken: {time_taken} seconds')