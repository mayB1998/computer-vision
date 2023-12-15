import cv2
import numpy as np

def put_text(image, org, text, color=(0, 0, 255), fontScale=0.7, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    if not isinstance(org, tuple):
        (label_width, label_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)
        org_w = 0
        org_h = 0

        h, w, *_ = image.shape

        place_h, place_w = org.split("_")

        if place_h == "top":
            org_h = label_height
        elif place_h == "bottom":
            org_h = h
        elif place_h == "center":
            org_h = h // 2 + label_height // 2

        if place_w == "left":
            org_w = 0
        elif place_w == "right":
            org_w = w - label_width
        elif place_w == "center":
            org_w = w // 2 - label_width // 2

        org = (org_w, org_h)

    image = cv2.putText(image, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image

def play_trip(l_frames, r_frames=None, lat_lon=None, timestamps=None, color_mode=False, playback_speed=10, win_name="Trip"):
    l_r_mode = r_frames is not None

    if not l_r_mode:
        r_frames = [None]*len(l_frames)

    frame_count = 0
    for i, frame_step in enumerate(zip(l_frames, r_frames)):
        img_l, img_r = frame_step

        if not color_mode:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            if img_r is not None:
                img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)


        if img_r is not None:
            img_l = put_text(img_l, "top_center", "Left")
            img_r = put_text(img_r, "top_center", "Right")
            show_image = np.vstack([img_l, img_r])
        else:
            show_image = img_l
        show_image = put_text(show_image, "top_left", "Press ESC to stop")
        show_image = put_text(show_image, "top_right", f"Frame: {frame_count}/{len(l_frames)}")

        if timestamps is not None:
            time = timestamps[i]
            show_image = put_text(show_image, "bottom_right", f"{time}")


        if lat_lon is not None:
            lat, lon = lat_lon[i]
            show_image = put_text(show_image, "bottom_left", f"{lat}, {lon}")

        cv2.imshow(win_name, show_image)

        key = cv2.waitKey(playback_speed)
        if key == 27:  # ESC
            break
        frame_count += 1
    cv2.destroyWindow(win_name)