# module yolov5model

# cv, data processing
import cv2
import torch
import yolov5

# custom
from auxiliary import *


def get_computation_device() -> str:
    """Returns available computation device

    Returns:
        str: 'cuda' or 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class YOLOv5Model:
    def __init__(
        self, 
        weights: str, 
        device: str = get_computation_device(),
        autoshape: bool = True,
        conf: float = 0.7,
        iou: float = 0.45,
        agnostic: bool = False,
        multi_label: bool = False,
        max_det: int = 100,
        verbose: bool = True
    ) -> None:
        """Creates and initalizes a yolov5 model

        Args:
            weights (str): model weights `*.pt`
            device (str, optional): 'cpu' or 'cuda'. Defaults to get_computation_device().
            autoshape (bool, optional): transform image to required shape upon passing to model. Defaults to True.
            conf (float, optional): model confidence. Defaults to 0.7.
            iou (float, optional): intersection over union threshold. Defaults to 0.45.
            agnostic (bool, optional): class agnostic without knowing what class an object belongs to. Defaults to False.
            multi_label (bool, optional): multiple labels per box. Defaults to False.
            max_det (int, optional): maximum detections per image. Defaults to 1000.
            verbose (bool, optional): verbose output. Defaults to True.
        """
        self._model = yolov5.load(
            model_path=weights,
            device=device,
            autoshape=autoshape,
            verbose=verbose
        )
        
        # init parameters
        self._model.conf = conf
        self._model.iou = iou
        self._model.agnostic = agnostic
        self._model.multi_label = multi_label
        self._model.max_det = max_det

    
    def predict(self, frame: cv2.Mat) -> tuple:
        """Predict model results

        Args:
            frame (cv2.Mat): image/video frame

        Returns:
            tuple: (number of objects found, list of detections each in tuples of ([left,top,w,h], confidence, detection_class))
        """
        results = self._model(frame)
        
        detections = list()
        for result in results.xyxy[0]:
            bbox = tuple(map(int, (result[0], result[1], result[2], result[3])))
            conf = result[4].item()
            detection_class = result[5].item()
            detections.append([bbox, conf, detection_class])
            
        return len(detections), detections


    def draw(self, frame: cv2.Mat, detections: list, color: tuple = Colors.LIME, thickness: int = 2, fontScale: float = 1, label_classes: list = None):
        """Draw bounding boxes

        Args:
            frame (cv2.Mat): image/video frame
            detections (list, optional): expected to be a list of detections, each in tuples of ([left,top,w,h], confidence, detection_class)
            color (tuple, optional): rgb color. Defaults to Colors.LIME.
            thickness (int, optional): stroke thickness. Defaults to 2.
            fontScale (float, optional): font scale. Defaults to 1.
            label_classes (list, optional): list string labels in the same order as your training data. Defaults to None.
        """
        if detections is None:
            return
        
        # convert color
        color = rgb2bgr(color)
        
        # visualize results
        for det in detections:
            x1, y1, x2, y2 = det[0]
            cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness)
            
            if label_classes:
                label_idx = int(det[5])
                cv2.putText(
                    img=frame,
                    text=label_classes[label_idx],
                    org=(int(x1 + (x2 - x1) / 10), int(y1 + (y2 - y1) / 3)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fontScale,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )
        
        
