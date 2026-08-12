import cv2
import numpy as np
import time
from collections import deque


COLOR_PALETTE = {
    "Red":     (0, 0, 255),
    "Green":   (0, 255, 0),
    "Blue":    (255, 0, 0),
    "Yellow":  (0, 255, 255),
    "Cyan":    (255, 255, 0),
    "Magenta": (255, 0, 255),
    "White":   (255, 255, 255),
    "Orange":  (0, 128, 255),
    "Gray":    (128, 128, 128),
    "Black":   (0, 0, 0)
}

HSV_RANGES = {
    "Red":     [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "Green":   [((36, 50, 50), (89, 255, 255))],
    "Blue":    [((90, 50, 50), (128, 255, 255))],
    "Yellow":  [((20, 100, 100), (30, 255, 255))],
    "Cyan":    [((80, 100, 100), (100, 255, 255))],
    "Magenta": [((140, 50, 50), (160, 255, 255))],
    "White":   [((0, 0, 200), (180, 30, 255))],
    "Orange":  [((10, 100, 100), (25, 255, 255))],
    "Gray":    [((0, 0, 50), (180, 50, 200))],
    "Black":   [((0, 0, 0), (180, 255, 30))]
}



def draw_label(img, text, position, text_color=(255, 255, 255), bg_color=(0, 0, 0), scale=0.5, thickness=1, padding=4):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    x, y = position
    
    bg_top_left = (x - padding, y - text_h - padding)
    bg_bottom_right = (x + text_w + padding, y + baseline + padding)
    
    
    cv2.rectangle(img, bg_top_left, bg_bottom_right, bg_color, thickness=-1)
    

    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def get_color_mask(hsv_frame, color_name):
    
    ranges = HSV_RANGES[color_name]
    mask = cv2.inRange(hsv_frame, ranges[0][0], ranges[0][1])
    if len(ranges) > 1:
        mask2 = cv2.inRange(hsv_frame, ranges[1][0], ranges[1][1])
        mask = cv2.bitwise_or(mask, mask2)
    return mask


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    show_labels = True
    frame_times = deque(maxlen=30)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        frame_times.append(curr_time - prev_time)
        prev_time = curr_time
        fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0.0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        
        overlay = frame.copy()

        opaque_shapes = []
        label_requests = []

        for color_name, color_bgr in COLOR_PALETTE.items():
            mask = get_color_mask(hsv, color_name)
            
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 800:  
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                label_requests.append((f"Color: {color_name} | Area: {int(area)}", (cx, cy - 15)))

                
                if color_name == "Red":
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    opaque_shapes.append(lambda f, c=color_bgr, center=(int(x), int(y)), r=int(radius): 
                                         cv2.circle(f, center, r, c, thickness=2))

                elif color_name == "Green":
                    x, y, w, h = cv2.boundingRect(cnt)
                    opaque_shapes.append(lambda f, c=color_bgr, pt1=(x, y), pt2=(x + w, y + h): 
                                         cv2.rectangle(f, pt1, pt2, c, thickness=2))

                elif color_name == "Blue":
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, thickness=-1)

                elif color_name == "Yellow":
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        opaque_shapes.append(lambda f, c=color_bgr, e=ellipse: 
                                             cv2.ellipse(f, e, c, thickness=2))

                elif color_name == "Cyan":
                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    opaque_shapes.append(lambda f, c=color_bgr, pts=[approx]: 
                                         cv2.polylines(f, pts, isClosed=True, color=c, thickness=2))

                elif color_name == "Magenta":
                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    cv2.fillPoly(overlay, [approx], color_bgr)

                elif color_name == "White":
                    opaque_shapes.append(lambda f, c=color_bgr, center=(cx, cy): 
                                         cv2.arrowedLine(f, (center[0] - 40, center[1] - 40), center, c, thickness=2, tipLength=0.3))

                elif color_name == "Orange":
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    cv2.circle(overlay, (int(x), int(y)), int(radius), color_bgr, thickness=-1)

                elif color_name == "Gray":
                    opaque_shapes.append(lambda f, c=color_bgr, center=(cx, cy): (
                        cv2.line(f, (center[0] - 15, center[1]), (center[0] + 15, center[1]), c, thickness=2),
                        cv2.line(f, (center[0], center[1] - 15), (center[0], center[1] + 15), c, thickness=2)
                    ))

                elif color_name == "Black":
                    x, y, w, h = cv2.boundingRect(cnt)
                    opaque_shapes.append(lambda f, c=color_bgr, pt1=(x, y), pt2=(x + w, y + h): 
                                         cv2.rectangle(f, pt1, pt2, c, thickness=2, lineType=cv2.LINE_AA))

    
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

   
        for shape_func in opaque_shapes:
            shape_func(frame)


        if show_labels:
            for text, pos in label_requests:
                draw_label(frame, text, pos, text_color=(255, 255, 255), bg_color=(20, 20, 20), scale=0.45)

        
        draw_label(frame, f"FPS: {fps:.1f}", (10, 25), text_color=(0, 255, 0), bg_color=(0, 0, 0), scale=0.55, thickness=2)
        toggle_status = "ON" if show_labels else "OFF"
        draw_label(frame, f"Labels ('t'): {toggle_status}", (10, 50), text_color=(255, 255, 255), bg_color=(0, 0, 0), scale=0.55, thickness=1)

        cv2.imshow("Multi-Color Shape Tracker HUD", frame)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            show_labels = not show_labels  

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()