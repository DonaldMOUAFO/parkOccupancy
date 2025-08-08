import argparse
from src.config import config
from src.domain.generate_coordinates import CoordinateGenerator
from src.domain.corresponding_points import PointsCorrespondance
from src.domain.homography_estimation import DetectFeature

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--image", type=str, required=False, 
        default=config.camera_frame_path,
        dest="image_path",
        help="Provide the image path to locate parking place on"               
    )

    parser.add_argument(
        "-c", "--collect", type=str, required=False, 
        default="True",
        dest="collect",
        help="Collect corresponding points for Homography estimation"               
    )

    parser.add_argument(
        "-sp", "--save_pts", type=str, required=False, 
        default="False",
        dest="save_pts",
        help="Save the collected corresponding points to a file"               
    )

    parser.add_argument(
        "-hm", "--homography", type=str, required=False, 
        default="False",
        dest="homography",
        help="Estimate homagraphy matrix from the collected points"               
    )

    parser.add_argument(
        "-cc", "--slot_coordinates", type=str, required=False, 
        default="False",
        dest="collect_slot_coordinates",
        help="Collect parking slot coordinates for the camera view"               
    )

    return parser.parse_args()


def main() :

    args = parse_args()

    camera_view_image_path   = config.camera_image_view_path
    bird_eye_view_image_path = config.bird_view_path
    print(f"args.save_pts= {args.save_pts}")
    GeneratePoints = PointsCorrespondance(camera_view_image_path, bird_eye_view_image_path, args.save_pts)

    if args.collect.lower() == 'true':
        
        GeneratePoints.collect_coordinates()

    if args.homography.lower() == "true":
        Transformator = DetectFeature(method="sift", from_pts=True)
        H, matches, kp1, kp2, mask = Transformator.compute_homography()
        Transformator.save_homography(H, config.HOMAGRAPHY_MATRIX_PATH)
        print("Homography matrix H:")
        print(H)
    if args.collect_slot_coordinates.lower() == "true":
        SlotCoordinateGenerator = CoordinateGenerator(args.image_path)
        SlotCoordinateGenerator.generate_coordinate()


if __name__ == "__main__" :
    main()

# python main.py -c false -sp false -hm true -cc True -i ../../data/camera_frame.png

# To do : implement the coordinate generator save confirmation.