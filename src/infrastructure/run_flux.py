import cv2
from datetime import datetime

# Open video file (replace with 0 for webcam)
# cap = cv2.VideoCapture("/home/besttic-rd/Documents/besttic/parkOccupancy/data/2020-03-19-08-00-00_scene_0013_BB_write.avi")
cap = cv2.VideoCapture(
    #"/home/besttic-rd/Vidéos/vlc-record-2025-09-12-11h39m43s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"
    #"../../data/scale/vlc-record-2025-09-05-14h03m19s-rtsp_10.210.19.177_554_0_profile2_media.smp-.mp4"


    #"/home/besttic-rd/Vidéos/vlc-record-2025-09-12-11h53m42s-rtsp___10.210.19.177_554_1_profile2_media.smp-.mp4"
    #"/home/besttic-rd/Vidéos/vlc-record-2025-09-12-11h53m42s-rtsp___10.210.19.177_554_1_profile2_media.smp-.mp4"
    #"../../data/scale/vlc-record-2025-09-05-14h16m20s-rtsp_10.210.19.177_554_1_profile2_media.smp-.mp4"

    #"../../data/scale/vlc-record-2025-09-05-14h18m43s-rtsp_10.210.19.177_554_2_profile2_media.smp-.mp4"
    "/home/besttic-rd/Vidéos/vlc-record-2025-09-19-10h02m09s-rtsp___10.210.19.177_554_2_profile2_media.smp-.mp4"

    #"/home/besttic-rd/Téléchargements/2020-03-19-08-00-00_scene_0016_BB_write.avi"
    #"/home/besttic-rd/Téléchargements/2020-03-20-08-00-00.scene_0001_BB_write.avi"
    #"/home/besttic-rd/Téléchargements/2020-03-19-08-00-00_scene_0018_BB_write.avi"
    )

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot fetch the frame.")
        break

    # Show frame
    cv2.imshow("Video", frame)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # write frame to file
    if cv2.waitKey(1) == ord('s'):
        #cv2.imwrite(f"../../data/scale/frame_focale_0_{time_str}.jpg", frame)
        #cv2.imwrite("../../data/scale/frame_focale_1_{time_str}.jpg", frame)
        #cv2.imwrite("../../data/scale/frame_focale_2_{time_str}.jpg", frame)
        #cv2.imwrite(f"../../data/camera_frame_focal_1_{time_str}.jpg", frame)
        cv2.imwrite(f"../../data/camera_frame_focal_2_{time_str}.jpg", frame)
        #cv2.imwrite("../../data/camera_frame_3_{time_str}.png", frame)

        break

    # Quit on 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
