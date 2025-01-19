from scipy.io import loadmat, savemat
import numpy as np
import os
import fnmatch
from pyflann import FLANN
import sys
from graph_search import uniform_cost_search

# Calculates homography from the second image to the first
def calculateHomography(mappings):
    number_of_points = mappings.shape[0]

    H = np.zeros((3, 3))
    A = np.zeros((number_of_points * 2, 8))
    B = np.zeros((number_of_points * 2, 1))

    for i in range(number_of_points):
        A[i*2, 0] = mappings[i, 2]
        A[i*2, 1] = mappings[i, 3]
        A[i*2, 2] = 1
        A[i*2, 3] = 0
        A[i*2, 4] = 0
        A[i*2, 5] = 0
        A[i*2, 6] = - mappings[i, 2] * mappings[i, 0]
        A[i*2, 7] = - mappings[i, 3] * mappings[i, 0]

        A[i*2+1, 0] = 0
        A[i*2+1, 1] = 0
        A[i*2+1, 2] = 0
        A[i*2+1, 3] = mappings[i, 2]
        A[i*2+1, 4] = mappings[i, 3]
        A[i*2+1, 5] = 1
        A[i*2+1, 6] = - mappings[i, 2] * mappings[i, 1]
        A[i*2+1, 7] = - mappings[i, 3] * mappings[i, 1]

        B[i*2, 0] = mappings[i, 0]
        B[i*2+1, 0] = mappings[i, 1]   
    

    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    H[0, 0] = X[0][0]
    H[0, 1] = X[1][0]
    H[0, 2] = X[2][0]
    H[1, 0] = X[3][0]
    H[1, 1] = X[4][0]
    H[1, 2] = X[5][0]
    H[2, 0] = X[6][0]
    H[2, 1] = X[7][0]
    H[2, 2] = 1

    return H

def applyHomography(H_matrix, original_points):
    original_points = np.hstack((original_points, np.ones((original_points.shape[0], 1))))

    transformed_points = original_points @ H_matrix.T

    transformed_points /= transformed_points[:, 2].reshape(-1, 1)

    return transformed_points[:, 0:2]


# Finds homography between two images (from the second image to the first). First image is reference.
def findHomographyBetweenImages(path_to_keypoints_1,
                                   path_to_keypoints_2,
                                   number_of_best_matches = 1000,
                                   ransac_iterations =1000,
                                   threshold = 3,
                                   minimum_number_of_inliers = 100):

    flann = FLANN()

    image_1_keypoints = loadmat(path_to_keypoints_1)['kp']
    image_1_descriptors = loadmat(path_to_keypoints_1)['desc']

    image_2_keypoints = loadmat(path_to_keypoints_2)['kp']
    image_2_descriptors = loadmat(path_to_keypoints_2)['desc']

    # Configure FLANN parameters
    params = flann.build_index(image_2_descriptors, algorithm='kdtree', trees=5)

    # Perform nearest neighbor search
    # Returns indices of nearest neighbors in des2 and their distances
    indices, dists = flann.nn_index(image_1_descriptors, num_neighbors=1)

    # Pair each vector in array1 with its closest vector in array2
    pairs = np.column_stack((np.arange(image_1_descriptors.shape[0]), indices))

    # Sort pairs by distance
    sorted_indices = np.argsort(dists)[:number_of_best_matches]
    best_matches = pairs[sorted_indices]

    # Make array with keypoints
    keypoint_pairs = np.column_stack((image_1_keypoints[best_matches[:, 0]], image_2_keypoints[best_matches[:, 1]]))

    # RANSAC
    best_homography = np.zeros((3, 3))
    best_inliers = np.empty((1))
    best_number_of_inliers = 0

    for i in range(ransac_iterations):
        rng = np.random.default_rng()
        random_matches = rng.choice(keypoint_pairs, 4, replace=False)

        homography = calculateHomography(random_matches)

        result_from_the_model = applyHomography(homography, keypoint_pairs[:, 2:4])
        result_from_the_dataset = keypoint_pairs[:, 0:2]

        distance = np.linalg.norm(result_from_the_dataset - result_from_the_model, axis=1)

        inliers = distance < threshold
        number_of_inliers = np.sum(inliers)

        if number_of_inliers > best_number_of_inliers:
            best_inliers = inliers
            best_number_of_inliers = number_of_inliers
            best_homography = homography

    if(best_number_of_inliers > minimum_number_of_inliers):
        # Recompute homography
        best_keypoints = keypoint_pairs[best_inliers]
        best_homography = calculateHomography(best_keypoints)

        outliers_ratio = (keypoint_pairs.shape[0] - best_number_of_inliers) / keypoint_pairs.shape[0]

        return 1, best_homography, best_number_of_inliers, outliers_ratio
    else:
        return 0, best_homography, 0, []


def calculateComposedHomography(direct_homographies, initial_image, path):
    current_image = initial_image
    H = np.eye(3)
    for step in path:
        H = direct_homographies[(step, current_image)][0] @ H
        current_image = step

    return H

def calculateHomographyBetweenNodes(direct_homographies, calculated_homographies, reference_frame, frame):
    path, cost = uniform_cost_search(direct_homographies, frame, reference_frame)
    if path is not None:
        path = path[1:]
    else:
        return calculated_homographies[:, :, frame-2]
    return calculateComposedHomography(direct_homographies, frame, path)

def getInputOutputFolders(video_indices, input_output_pairs, image_number):
    for i, index in enumerate(video_indices):
        if image_number <= index:
            return i, input_output_pairs[i][0], input_output_pairs[i][1]

##########################################################

if __name__ == "__main__":

    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print("Usage: python main.py ref_dir input1_dir output1_dir [input2_dir output2_dir ...]")
        sys.exit(1)

    # Parse arguments
    ref_directory = sys.argv[1]
    input_output_pairs = sys.argv[2:]  # Remaining arguments

    # Group into input-output pairs
    input_output_pairs = [(input_output_pairs[i], input_output_pairs[i + 1])
                          for i in range(0, len(input_output_pairs), 2)]

    image_file_pattern = "img_*.jpg"

    # Adding reference image
    images = [(ref_directory + '/' + file) for file in os.listdir(ref_directory) if fnmatch.fnmatch(file, image_file_pattern)]

    number_of_videos = len(input_output_pairs)
    video_indices = []    
    for input_dir, output_dir in input_output_pairs:
        images_from_video = [(input_dir + '/' + file) for file in os.listdir(input_dir) if fnmatch.fnmatch(file, image_file_pattern)]
        number_of_frames = len(images_from_video)

        # Store the[-1] is not None number of last frame in the video
        if video_indices:
            video_indices.append(video_indices[-1] + number_of_frames)
        else:
            video_indices.append(number_of_frames)

        images.extend(sorted(images_from_video))

    direct_homographies = {}
    
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            image_1 = images[i].replace('img_', 'kp_')
            image_1 = image_1.replace('jpg', 'mat')
            
            image_2 = images[j].replace('img_', 'kp_')
            image_2 = image_2.replace('jpg', 'mat')

            res, H, number_of_inliers, ratio = findHomographyBetweenImages(image_1,
                                image_2,
                                minimum_number_of_inliers=50,
                                ransac_iterations=2000,
                                threshold=3
                                )
            
            if(res):
                direct_homographies[(i, j)] = (H, ratio)
                direct_homographies[(j, i)] = (np.linalg.inv(H), ratio)

    homographies = np.zeros((3, 3, len(images)-1))
    for number_of_image in range(1, len(images)):
        homographies[:, :, number_of_image-1] = calculateHomographyBetweenNodes(direct_homographies, homographies, 0, number_of_image)

        index_of_folder, input_dir, output_dir = getInputOutputFolders(video_indices, input_output_pairs, number_of_image)

        yolo_file_number = number_of_image if not index_of_folder else (number_of_image-video_indices[index_of_folder-1])

        yolo_file = f"{input_dir}/yolo_{yolo_file_number:04d}.mat"
        if os.path.exists(yolo_file):
            yolo_detection = loadmat(yolo_file)

            ids = yolo_detection['id']
            classes = yolo_detection['class']
            xyxys = yolo_detection['xyxy']

            blcs = xyxys[:, 0:2]
            trcs = xyxys[:, 2:4]

            blcs_transformed = applyHomography(homographies[:, :, number_of_image-1], blcs)
            trcs_transformed = applyHomography(homographies[:, :, number_of_image-1], trcs)

            yolo_transformed = np.hstack((blcs_transformed, trcs_transformed))


            dictionary = {'class': classes, 'id': ids, 'xyxy': yolo_transformed}
            os.makedirs(output_dir, exist_ok=True) 
            savemat(f"{output_dir}/yolooutput_{yolo_file_number:04d}.mat", dictionary)

    starting_index = 0
    for index in video_indices:
        homographies_for_video = homographies[:, :, starting_index : (index)]
        starting_index = index

        index_of_folder, input_dir, output_dir = getInputOutputFolders(video_indices, input_output_pairs, index)

        dictionary = {'H': homographies_for_video}
        os.makedirs(output_dir, exist_ok=True) 
        savemat(f"{output_dir}/homographies.mat", dictionary)