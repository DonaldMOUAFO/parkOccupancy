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
        "-cpc", "--collect_place_coordinates", type=str, required=False, 
        default="False",
        dest="collect_parking_coordinates",
        help="Collect parking slot coordinates for the camera view"               
    )

    parser.add_argument(
        "-spc", "--save_places_coordinates", type=str, required=False, 
        default="True",
        dest="save_places_coordinates",
        help="save the collected parking place coordinates to a file"               
    )

    parser.add_argument(
        "-c", "--collect", type=str, required=False, 
        default="False",
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
        "-pp", "--parking_places_file_path", type=str, required=False, 
        default=config.PARKING_FILE_PATH,
        dest="parking_places_file_path",
        help="save the collected slot_coordinates"               
    )

    return parser.parse_args()

def main() :

    args = parse_args()
    print(
        f"""args.image_path={args.image_path}\n args.collect={args.collect}\n args.save_pts={args.save_pts}\n 
            args.homography={args.homography}\n args.collect_parking_coordinates={args.collect_parking_coordinates}n\n
            args.save_place_coordinates={args.save_places_coordinates}\n args.place_file_path={args.parking_places_file_path}"""
    )

    if args.collect.lower() == 'true':

        camera_view_image_path   = config.camera_image_view_path
        bird_eye_view_image_path = config.bird_view_path

        GeneratePoints = PointsCorrespondance(camera_view_image_path, bird_eye_view_image_path, args.save_pts)
        GeneratePoints.collect_coordinates()

    if args.homography.lower() == "true":
        Transformator = DetectFeature(method="sift", from_pts=True)
        H, matches, kp1, kp2, mask = Transformator.compute_homography()
        Transformator.save_homography(H, config.HOMAGRAPHY_MATRIX_PATH)
        print("Homography matrix H:")
        print(H)

    if args.collect_parking_coordinates.lower() == "true":
        SlotCoordinateGenerator = CoordinateGenerator(args.image_path)
        SlotCoordinateGenerator.generate_coordinate(args.save_places_coordinates, args.parking_places_file_path)


if __name__ == "__main__" :
    main()

# python main.py -c false -sp false -hm true -cpc True -i ../../data/camera_frame.png   for experiemental image
# python main.py -c false -sp false -hm true -cpc True -i ../../data/scale/frame_focale_0.jpg
# python main.py -c false -sp false -hm true -cpc True -sc True -i ../../data/camera_frame.png -pp ../../data/scale/parking_place_coordinate_focale_0.json
# python main.py -c false -sp false -hm true -cpc True -spc True -i ../../data/scale/frame_focale_0.jpg -pp ../../data/scale/parking_place_coordinate_focale_0.json

#python main.py -c false -sp false -hm true -cpc True -spc True -i ../../data/scale/frame_focale_0.jpg -pp ../../data/scale/parking_place_coordinate_focale_0_1.jsonclea
# To do : implement the coordinate generator save confirmation.