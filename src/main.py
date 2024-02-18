# module main

# cv, data processing
import cv2

# custom
from auxiliary import *
from yolov5model import *


# init
model = YOLOv5Model(weights='assets/weights/pedestrian_yolov5m_p79.pt', conf=0.5)


def update(params):
    frame = params['frame']
    
    # detect
    nobj, detections = model.predict(frame)
    model.draw(frame, detections)
    
    # process events
    key = cv2.waitKey(30)
    if (key & 0xFF == ord('h')) or key == 40:
        params['hide_debug_info'] = not params['hide_debug_info']
        
    # update params
    params.update({'nobj': nobj})
    
    # draw                    
    draw(frame, params)

    return frame


def draw(frame: cv2.Mat, params: dict = None):
    # define data
    area = 25
    data = [
        f'AREA: {area} m2',
        f'PEOPLE: {params["nobj"]}',
        f'DENSITY (PEOPLE/AREA): {(params["nobj"]/area):.2f} ppl/m2',
        f'>> "h" to hide debug information',
    ]
    
    # draw data
    if not params['hide_debug_info']:
        for i, entry in enumerate(data):
            cv2.putText(
                img=frame,
                text=entry,
                org=(15, 30*(i+1)),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
    

if __name__ == '__main__':
    update_params = {
        'area': 25,
        'hide_debug_info': False,
    }
    display_video('assets/test_media/test_mix_1.mp4', func=update, func_params=update_params, frame_rate=3, draw_fps=True)
