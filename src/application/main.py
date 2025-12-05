import argparse
from src.config import config
from src.domain.generate_coordinates import CoordinateGenerator
from src.domain.troubleshoot_occupancy_v3 import run_place_counting
from src.domain.corresponding_points import PointsCorrespondance
from src.domain.homography_estimation import DetectFeature
from ultralytics import YOLO

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
        default="False",
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
        "-pcf", "--pts_corsp_file", type=str, required=False, 
        default=config.points_correspondance_file,
        dest="pts_corsp_file",
        help="File path to save the collected corresponding points"               
    )

    parser.add_argument(
        "-fr", "--frame", type=str, required=False, 
        default="focale_0",
        dest="frame",
        help="Precise the Frame identifier to differentiate saving files"               
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

    parser.add_argument(
        "-ppc", "--parking_place_counting", type=str, required=False, 
        default="True",
        dest="parking_place_counting",
        help="Run parking place counting based on vehicle detection tracking and motion evaluation"               
    )

    return parser.parse_args()

def main() :

    args = parse_args()
    print(
        f"""args.image_path={args.image_path}\n args.collect={args.collect}\n args.save_pts={args.save_pts}\n 
            args.homography={args.homography}\n args.collect_parking_coordinates={args.collect_parking_coordinates}\n
            args.save_place_coordinates={args.save_places_coordinates}\n args.place_file_path={args.parking_places_file_path}"""
    )

    if args.collect.lower() == 'true':

        camera_view_image_path   = config.camera_image_view_path
        bird_eye_view_image_path = config.bird_view_path

        GeneratePoints = PointsCorrespondance(
            camera_view_image_path, 
            bird_eye_view_image_path,
            frame=args.frame,
            file_path=args.pts_corsp_file,
            save_pts=args.save_pts
        )
        GeneratePoints.collect_coordinates()

    if args.homography.lower() == "true":
        Transformator = DetectFeature(method="sift", from_pts=True)
        H, matches, kp1, kp2, mask = Transformator.compute_homography()
        Transformator.save_homography(H, args.frame, config.HOMAGRAPHY_MATRIX_PATH)
        print("Homography matrix H:")
        print(H)

    if args.collect_parking_coordinates.lower() == "true":
        SlotCoordinateGenerator = CoordinateGenerator(args.image_path)
        SlotCoordinateGenerator.generate_coordinate(
            args.save_places_coordinates, 
            args.parking_places_file_path
        )
    
    if args.parking_place_counting.lower() == "true":
        model = YOLO("yolo11l.pt")
        flux_url = config.prepare_flux_url(
            config.CAMERA_LOGIN_PATH,
            focal=args.frame
        )

        if int(args.frame) == 0:
            video_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"
            # video_path = "/home/besttic-rd/Vidéos/vlc-record-2025-09-05-14h03m19s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"
        elif int(args.frame) == 1:
            #video_path = "/home/besttic-rd/Vidéos/vlc-record-2025-09-12-11h53m42s-rtsp___10.210.19.177_554_1_profile2_media.smp-.mp4"
            video_path = "/home/besttic-rd/Vidéos/vlc-record-2025-09-05-14h16m20s-rtsp___10.210.19.177_554_1_profile2_media.smp-.mp4"
        elif int(args.frame) == 2:
            video_path = "/home/besttic-rd/Vidéos/vlc-record-2025-09-05-14h18m43s-rtsp___10.210.19.177_554_2_profile2_media.smp-.mp4" 
            
        run_place_counting(video_path=video_path, model=model, focal=args.frame)

if __name__ == "__main__" :
    main()

# python main.py -c false -sp false -hm true -cpc True -i ../../data/camera_frame.png   for experiemental image
# python main.py -c false -sp false -hm true -cpc True -i ../../data/scale/frame_focale_0.jpg
# python main.py -c false -sp false -hm true -cpc True -sc True -i ../../data/camera_frame.png -pp ../../data/scale/parking_place_coordinate_focale_0.json
# python main.py -c false -sp false -hm true -cpc True -spc True -i ../../data/scale/frame_focale_0.jpg -pp ../../data/scale/parking_place_coordinate_focale_0.json

#python main.py -c false -sp false -hm true -cpc True -spc True -i ../../data/scale/frame_focale_0.jpg -pp ../../data/scale/parking_place_coordinate_focale_0_1.json
# To do : implement the coordinate generator save confirmation.
# python troubleshoot_occupancy_v3.py 

# self.json_coordinate_format = {0: '[[665, 257], [737, 249], [593, 297], [544, 297]]',
#                                1: '[[543, 297], [597, 298], [425, 356], [397, 344]]', 
#                                2: '[[174, 417], [284, 409], [0, 528], [2, 486]]', 
#                                3: '[[927, 264], [1272, 264], [1252, 285], [892, 286]]', 
#                                4: '[[892, 286], [1252, 286], [1237, 303], [862, 301]]', 
#                                5: '[[862, 301], [1238, 303], [1219, 323], [830, 320]]', 
#                                6: '[[830, 320], [1219, 323], [1198, 349], [799, 338]]', 
#                                7: '[[798, 339], [1200, 349], [1175, 375], [764, 360]]', 
#                                8: '[[763, 360], [1176, 375], [1146, 405], [730, 379]]',
#                                9: '[[730, 379], [1145, 405], [1115, 438], [699, 397]]', 
#                                10: '[[667, 415], [1096, 460], [1059, 503], [612, 447]]', 
#                                11: '[[612, 447], [1059, 503], [1017, 551], [555, 478]]', 
#                                12: '[[555, 478], [1017, 551], [969, 605], [493, 515]]', 
#                                13: '[[493, 515], [967, 607], [910, 671], [422, 554]]', 
#                                14: '[[423, 554], [910, 672], [841, 748], [337, 604]]', 
#                                15: '[[337, 604], [842, 749], [760, 842], [252, 653]]', 
#                                16: '[[252, 653], [758, 843], [647, 969], [132, 720]]', 
#                                17: '[[132, 720], [646, 970], [414, 1078], [5, 792]]'}

# Homography matrix H:
# h = [[-6.32359385e-01 -1.87153625e+00  8.71131287e+02]
#  [-5.41868687e-01 -3.55696392e+00  1.20834229e+03]
#  [-9.59175639e-04 -4.76889824e-03  1.00000000e+00]]