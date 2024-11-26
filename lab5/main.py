from utils import *


image_path = 'images/0AA0A2.jpg'
noFace_image_path = 'otherImages/nature.jpg'
image_dir = 'images'
results_dir = 'results'
labels_file = 'labels.csv'
split_data_dir = 'splitData'

if __name__ == '__main__':
    # Task 1 - Blur an image, Sharpen an image
    blur_image(image_path, results_dir)
    sharpen_image(image_path, results_dir)

    # Task 2 - Detect faces in an image
    face_coordinates = detect_face_image(image_path, results_dir)
    if face_coordinates:
        print(f"Face detected for {image_path} at coordinates: {face_coordinates}")
    else:
        print(f"No face detected in the image {image_path}.")

    # Detect faces in an image with no face    
    face_coordinates = detect_face_image(noFace_image_path, results_dir)
    if face_coordinates:
        print(f"Face detected for {noFace_image_path} at coordinates: {face_coordinates}")
    else:
        print(f"No face detected in the image {noFace_image_path}.")  

    # Task 3 - Detect whether a photo is accepted for a passport
    is_valid = valid_passport_image(image_path) 

    # Task 4 - Split data into training, validation and testing sets
    if not os.path.exists(f'{split_data_dir}' or not os.listdir(split_data_dir)):
        split_data(labels_file, split_data_dir)
    total_files = os.listdir(image_dir)
    train_files = 0
    for root, _, files in os.walk(f'{split_data_dir}/train'):
        train_files += len(files)
    val_files = 0
    for root, _, files in os.walk(f'{split_data_dir}/validation'):
        val_files += len(files)
    test_files = 0
    for root, _, files in os.walk(f'{split_data_dir}/test'):
        test_files += len(files)
    print(f"\nTotal number of images: {len(total_files)}")
    print(f"Train set: {train_files} images")
    print(f"Validation set: {val_files} images")
    print(f"Test set: {test_files} images")

    # Task 5 - Train a CNN model to classify images
    trained_model, training_history = train_cnn_model(split_data_dir, batch_size=8, epochs=10, learning_rate=0.00001)

    # Task 6 - Evaluate the CNN vs OpenCV 
    correct_detected = evaluate_openCv(split_data_dir)
    accuracy = correct_detected / test_files * 100
    print(f"\nAccuracy of OpenCV: {accuracy:.2f}%")
