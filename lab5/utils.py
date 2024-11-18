import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from shutil import copyfile


def blur_image(image_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Convert the image from BGR to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    k_size = (19, 19)
    blurred_image = cv2.blur(image_rgb, k_size) 

    # Plot the initial and blurred images using subplots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(blurred_image)
    plt.title('Blurred Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_path}/originalVsBlurred.png', bbox_inches='tight')
    plt.close()

def sharpen_image(image_path, save_path):
    # Read the image from the specified
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    
    # Convert the image from BGR to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # This is basically "laplacian" filter (8 -neighnors). It enhances the edges, corners etc. high frequency regions.
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharpen_image = cv2.filter2D(image_rgb, -1, kernel)

    # Plot the initial and sharpened images using subplots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(sharpen_image)
    plt.title('Sharpened Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_path}/originalVsSharpened.png', bbox_inches='tight')
    plt.close()

def detect_face_image(image_path, save_path):
    # Read the image from the specified
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )  
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    # If no faces are detected, return None
    if len(face) == 0:
        return None
    x, y, w, h = face[0]
    
    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

    # Plot the image with face detected
    plt.figure(figsize=(10, 5))  
    plt.imshow(img_rgb)
    plt.title('Detected Face')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_path}/detectedFace.png', bbox_inches='tight')

    x, y, w, h = int(x), int(y), int(w), int(h)
    return (x, y, w, h)

def is_color_image(image):
    # Check if the image is in color by comparing RGB values of all pixels
    if np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 1], image[..., 2]):
        return False
    return True

def is_portrait_or_square(image):
    # Check if the image is in portrait orientation or square
    height, width = image.shape[:2]
    return height >= width

def detect_face(image):
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the image to grayscale (Haar Cascade works on grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    return faces

def are_eyes_at_same_level(image, face):
    # Load the pre-trained Haar Cascade classifier for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # Extract the face region from the image
    x, y, w, h = face
    face_region = image[y:y+h, x:x+w]
    # Convert the face region to grayscale
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    # Check if two eyes are detected
    if len(eyes) != 2:
        return False
    # Get the y-coordinates of the eyes
    eye1_y = eyes[0][1]
    eye2_y = eyes[1][1]
    # Check if the eyes are at the same level (with a max. error of 5 pixels)
    return abs(eye1_y - eye2_y) <= 5

def is_head_size_valid(image, face):
    # Check if the head represents 20% to 50% of the area of the photo
    image_area = image.shape[0] * image.shape[1]
    face_area = face[2] * face[3]
    face_area_percentage = (face_area / image_area) * 100
    return 20 <= face_area_percentage <= 50

def valid_passport_image(image_path):
    # Read the image from the specified path
    image = cv2.imread(image_path)
    if image is None:
        # print(f"Error: Unable to load image from {image_path}")
        return False
    # Check if the photo is in color
    if not is_color_image(image):
        # print("The photo is not in color.")
        return False
    # Check if the photo is in portrait orientation or square
    if not is_portrait_or_square(image):
        # print("The photo is not in portrait orientation or square.")
        return False
    # Detect faces in the image
    faces = detect_face(image)
    if len(faces) != 1:
        # print("The photo does not contain exactly one person or none at all.")
        return False
    # Check if the eyes of the subject are at the same level
    if not are_eyes_at_same_level(image, faces[0]):
        # print("The eyes of the subject are not at the same level.")
        return False
    # Check if the head size is valid
    if not is_head_size_valid(image, faces[0]):
        # print("The head size is not valid.")
        return False

    # print("\nThe photo is accepted for a passport.")
    return True

def split_data(labels_file, output_folder):
    """
    Splits data into train, validation, and test sets and organizes them into
    class_0 and class_1 folders based on labels. Ensures that the difference in the number of 
    images between class_0 and class_1 is at most 3, and applies augmentation for training images.
    Keeps both original and augmented images.

    Parameters:
        labels_file (str): Path to the CSV file containing file paths and labels.
        output_folder (str): Directory where the split data will be stored.
    """
    # Split ratios
    train_ratio = 0.80
    val_ratio = 0.10
    test_ratio = 0.10

    # Create directories for split data
    split_dirs = ['train', 'validation', 'test']
    classes = ['class_0', 'class_1']  # Folders for binary classification
    for split_dir in split_dirs:
        for cls in classes:
            os.makedirs(os.path.join(output_folder, split_dir, cls), exist_ok=True)

    # Load labels CSV
    labels = pd.read_csv(labels_file)

    # Split the data into train, validation, and test
    train, temp = train_test_split(labels, test_size=(1 - train_ratio), random_state=46, stratify=labels['label'])
    val, test = train_test_split(temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=46, stratify=temp['label'])

    # Helper function to balance classes with a max difference of 3 samples
    def balance_classes(dataframe):
        # Count instances of each class
        class_0_count = len(dataframe[dataframe['label'] == 0])
        class_1_count = len(dataframe[dataframe['label'] == 1])

        # Calculate the difference in size
        size_diff = abs(class_0_count - class_1_count)

        # Downsample the larger class to minimize the difference to 3
        if size_diff > 2:
            if class_0_count > class_1_count:
                class_0 = dataframe[dataframe['label'] == 0].sample(class_1_count + 2, random_state=46)
                class_1 = dataframe[dataframe['label'] == 1]
            else:
                class_1 = dataframe[dataframe['label'] == 1].sample(class_0_count + 2, random_state=46)
                class_0 = dataframe[dataframe['label'] == 0]
        else:
            class_0 = dataframe[dataframe['label'] == 0]
            class_1 = dataframe[dataframe['label'] == 1]

        # Concatenate the balanced classes
        return pd.concat([class_0, class_1])

    # Balance the classes for each split
    train = balance_classes(train)
    val = balance_classes(val)
    test = balance_classes(test)

    # Helper function to copy files to the split directories
    def copy_files(dataframe, split_name, augment=False):
        for _, row in dataframe.iterrows():
            src_path = row['new_path']
            class_folder = 'class_1' if row['label'] else 'class_0'
            dest_path = os.path.join(output_folder, split_name, class_folder, os.path.basename(src_path))
            
            # Copy the original image first
            copyfile(src_path, dest_path)
            
            # If augmenting, apply augmentation
            if augment and split_name == 'train':
                # Load and convert the image to numpy array
                img = load_img(src_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)  # Reshape for flow
                
                # Define augmentation transformations
                datagen = ImageDataGenerator(
                    rotation_range=5, 
                    width_shift_range=0.1, 
                    height_shift_range=0.1, 
                    shear_range=0.1, 
                    zoom_range=0.2, 
                    horizontal_flip=False, 
                    fill_mode='nearest'
                )
                
                # Generate and save augmented images
                i = 0
                for _ in datagen.flow(x, batch_size=1, save_to_dir=os.path.dirname(dest_path), save_prefix='aug_', save_format='jpeg'):
                    i += 1
                    if i > 2:  # Generate a few augmentations per image
                        break

    # Copy original and augmented images to respective folders
    copy_files(train, 'train', augment=True)
    copy_files(val, 'validation', augment=False)
    copy_files(test, 'test', augment=False)

    print("Data has been successfully split and copied into class_0 and class_1 folders with both original and augmented images.")

def train_cnn_model(data_dir, image_size=(128, 128), batch_size=16, epochs=10, learning_rate=0.001):
    """
    Trains a CNN model for binary classification on passport photos.

    Parameters:
        data_dir (str): Path to the root directory containing 'train', 'validation', and 'test' subdirectories.
        image_size (tuple): Target size to which all images will be resized.
        batch_size (int): Number of samples per batch.
        epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model (tf.keras.Model): The trained CNN model.
        history (History): Training history containing loss and accuracy.
    """
    # Define paths
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    # Data generators for preprocessing
    datagen_args = {
        'rescale': 1.0 / 255,
        'rotation_range': 20,         
        'width_shift_range': 0.2,     
        'height_shift_range': 0.2,    
        'shear_range': 0.2,          
        'zoom_range': 0.2,           
        'horizontal_flip': True,     
        'fill_mode': 'nearest'        
    }
    train_gen = ImageDataGenerator(**datagen_args).flow_from_directory(
        train_dir, target_size=image_size, batch_size=batch_size, class_mode='binary'
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        val_dir, target_size=image_size, batch_size=batch_size, class_mode='binary'
    )
    test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        test_dir, target_size=image_size, batch_size=batch_size, class_mode='binary'
    )
    # Define CNN model
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Train the model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss * 100:.2f}%")

    return model, history

def evaluate_openCv(data_dir):
    correct = 0
    for root, _, files in os.walk(f'{data_dir}/test/class_1'):
        for file in files:
            if(valid_passport_image(f'{root}/{file}')):
                correct += 1
    for root, _, files in os.walk(f'{data_dir}/test/class_0'):
        for file in files:
            if(not valid_passport_image(f'{root}/{file}')):
                correct += 1           
    return correct        