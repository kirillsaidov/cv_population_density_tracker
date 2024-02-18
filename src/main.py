# module main

# system
import os
from datetime import datetime, timedelta

# cv, data processing
import cv2
import pandas as pd

# custom
from auxiliary import *
from yolov5model import *


# init
model = YOLOv5Model(weights='assets/weights/pedestrian_yolov5m_p79.pt', conf=0.5)


def log_density(csv_filename: str, dt: datetime, people: int, area: float):
    df = pd.DataFrame.from_dict({
        'datetime': dt,
        'people': [people],
        'area': [area],
        'density': [people/area],
    })
    
    # check if exists
    if os.path.exists(csv_filename):
        df_old = pd.read_csv(csv_filename)
        df = pd.concat([df_old, df])
    
    # save data
    df.to_csv(csv_filename, index=False)


def update(params: dict):
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
    
    # save density data
    dt_now = datetime.now()
    if dt_now - params['datetime'] > timedelta(seconds=10):
        log_density(params['csv'], dt_now, nobj, params['area'])
        params['datetime'] = dt_now

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
        'datetime': datetime.now(),
        'csv': 'population_density.csv',
        'hide_debug_info': False,
        'area': 25,
    }
    
    # run
    display_video(
        'assets/test_media/test_mix_1.mp4', 
        func=update, 
        func_params=update_params, 
        frame_rate=2, 
        draw_fps=True, 
        frame_size=(720, 480)
    )
