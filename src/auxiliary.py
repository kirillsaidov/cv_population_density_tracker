# module auxiliary

# system
import os
import time

# cv, data processing
import cv2


class Colors:
    """RGB color values
    """
    BLACK   = (  0,   0,   0)
    WHITE   = (255, 255, 255)
    RED     = (255,   0,   0)
    ORANGE  = (255, 165,   0)
    CORAL   = (255, 127,  80)
    LIME    = (  0, 255,   0)
    BLUE    = (  0, 255,   0)
    BROWN   = (139,  69,  19)
    GOLD    = (255, 215,   0)
    YELLOW  = (255, 255,   0)
    CYAN    = (  0, 255, 255)
    MAGENTA = (255,   0, 255)
    SILVER  = (192, 192, 192)
    GRAY    = (128, 128, 128)
    MAROON  = (128,   0,   0)
    OLIVE   = (128, 128,   0)
    GREEN   = (  0, 128,   0)
    PURPLE  = (128,   0, 128)
    TEAL    = (  0, 128, 128)
    NAVY    = (  0,   0, 128)



def image_resize(image: cv2.Mat, width: int = None, height: int = None, fast: bool = True) -> cv2.Mat:
    """Resizes the image and picks the correct interpolation method

    Args:
        image (cv2.Mat): image in cv2.Mat representation
        width (int, optional): new desired image width. Defaults to None.
        height (int, optional): new desired image height. Defaults to None.
        fast (bool, optional): prefer speed to image quality when resizing. Defaults to True.

    Returns:
        cv2.Mat: resized image
    """
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)
        
    # pick the right interpolation method
    inter = None
    image_area_old, image_area_new = w * h, dim[0] * dim[1]
    if image_area_new > image_area_old:
        inter = cv2.INTER_LINEAR if fast else cv2.INTER_CUBIC
    else:
        inter = cv2.INTER_AREA

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def frame_rate_elapsed(time_prev: list, frame_rate: int) -> bool:
    """Controls frame rate

    Args:
        time_prev (list(int)): previously updated time in a list
        frame_rate (int): frame rate
    
    Note:
        `time_prev` is a list containing a single int value, which is the previous time. 
        It is updated automatically within the function.

    Returns:
        bool: True upon frame elapsed (move on to next frame)
    """
    time_elapsed = time.time() - time_prev[0]
    if time_elapsed > 1./frame_rate:
        time_prev[0] = time.time()
        return True
    
    return False


def display_video(
    video_path: str, 
    window_title: str = 'VIDEO', 
    func = None,
    func_params: dict = dict(),
    frame_rate: int = 24,
    frame_size: tuple = (1080, 720),
    verbose: bool = True,
    draw_fps: bool = True,
    record_file_name: str = None,
    frame_failed_read_limit: int = 4,
    mouse_event_callback = None,
):
    """Play video file

    Args:
        video_path (str): path to video file
        window_title (str, optional): window title. Defaults to 'VIDEO'.
        func (in cv2.Mat, ret cv2.Mat, optional): apply additional operations to each frame `func({'frame': my_frame})`. Defaults to None.
        frame_rate (int, optional): frame rate per second. Defaults to 24.
        frame_size (tuple, optional): resize frame. Defaults to (1080, 720).
        verbose (bool, optional): verbose output. Defaults to True.
        draw_fps (bool, optional): draw fps to frames. Defaults to True.
        record_file_name (str, optional): recording file name. Defaults to None.
        frame_failed_read_limit (int, optional): stop after N attempts. Defaults to 4.
        mouse_event_callback (Any, optional): mouse callback `function(event, x, y, flags, params)`. Defaults to None.
    
    Note:
        1. `func` takes in a dictionary as parameters.
        1.1. `func({'frame': my_frame})`
        2. If `record_file_name` is None, it does not record anything
    """
    assert os.path.exists(video_path), f"Video file does not exist: {video_path}"
    
    # open video capture
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Failed to open VideoCapture"
        
    # create video writer
    video_writer = None if record_file_name is None else cv2.VideoWriter(
        record_file_name, 
        cv2.VideoWriter_fourcc('M','J','P','G'), 
        24, 
        (frame_size[0], frame_size[1])
    )
    
    # set mouse callback
    if mouse_event_callback:
        cv2.setMouseCallback(window_title, mouse_event_callback)

    time_prev = [time.time()]
    time_real_fps = time.time()
    frame_failed_read_counter = 0
    frame_fps_counter = 0
    frame_real_fps = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if frame_rate_elapsed(time_prev, frame_rate): 
                frame_fps_counter = frame_fps_counter + 1
                           
                # resize frame to the desired size
                frame = image_resize(frame, width=frame_size[0], height=frame_size[1])
                
                # draw fps
                if draw_fps:
                    cv2.putText(
                        img=frame,
                        text=f'FPS: {frame_real_fps}',
                        org=tuple(map(int, (frame_size[0]*0.45, 0.05*frame_size[1]))),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )

                # apply function
                if func is not None:
                    if func_params is None:
                        func_params = dict()
                        
                    func_params.update({'frame': frame})
                    frame = func(func_params)
                
                # display the result
                cv2.imshow(window_title, frame)
                
                # write frame to video file
                if video_writer: video_writer.write(frame)
            else:
                cap.grab()
        else:
            frame_failed_read_counter = frame_failed_read_counter + 1
            if verbose: 
                print(f'Error ({window_title}): failed to read a frame ({frame_failed_read_counter}/{frame_failed_read_limit})!')
        
        # if failed read N frames, exit
        if frame_failed_read_counter >= frame_failed_read_limit:
            break
        
        # process events
        key = cv2.waitKey(30)
        if (key & 0xFF == ord('q')) or key == 27:
            break
            
        # calculate real fps
        time_real_fps_elapsed = time.time() - time_real_fps
        if time_real_fps_elapsed > 1.0:
            frame_real_fps = int(frame_fps_counter / time_real_fps_elapsed)
            
            # update
            frame_fps_counter = 0
            time_real_fps = time.time()
            
    # release resources
    cap.release()
    if video_writer: video_writer.release()
    cv2.destroyAllWindows()


def rgb2bgr(color: tuple) -> tuple:
    """Converts RGB to BGR

    Args:
        color (tuple): rgb color value

    Returns:
        tuple: bgr color value
    """
    return (color[2], color[1], color[0])


def draw_bbox(
    frame: cv2.Mat, 
    bbox: tuple, 
    label: str = None, 
    color: tuple = Colors.LIME, 
    thickness: int = 2,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1,
    line_type: int = cv2.LINE_AA
):
    """Draw bounding box and label text

    Args:
        frame (cv2.Mat): image/video frame
        bbox (tuple): bounding box (x1, y1, x2, y2)
        label (str, optional): label string. Defaults to None.
        color (tuple, optional): rgb color. Defaults to Colors.LIME.
        thickness (int, optional): stroke thickness. Defaults to 2.
        font (int, optional): font type. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): fonst size. Defaults to 1.
        line_type (int, optional): line type. Defaults to cv2.LINE_AA.
    """
    # convert rgb to bgr
    color = rgb2bgr(color)
    
    # draw bbox
    x1, y1, x2, y2 = map(int, (bbox[0], bbox[1], bbox[2], bbox[3]))
    cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness)

    # draw label
    if label:
        cv2.putText(
            img=frame,
            text=label,
            org=(x1, y1),
            fontFace=font,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=line_type
        )





