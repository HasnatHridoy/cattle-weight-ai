from ultralytics import YOLO
import cv2
import random
import numpy as np

""" Image processing """
""" ---------------------------------------------------------------- """

def image_pro(img, result_seg, bounding_box, inch_b_w, inch_b_h):

      def overlay(image, mask, color, alpha):

          colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
          colored_mask = np.moveaxis(colored_mask, 0, -1)
          masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
          image_overlay = masked.filled()

          image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

          return image_combined


      def plot_one_box(x, img, color=None, label=None, line_thickness=3):
          # Plots one bounding box on image img
          tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
          color = color or [random.randint(0, 255) for _ in range(3)]
          c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
          cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
          if label:
              tf = max(tl - 1, 1)  # font thickness
              t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
              c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
              cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
              cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


      # Working with mask processing

      class_names = models.names

      red_color = [255, 0, 0]     # Red color
      firoza_color = [52, 235, 128]

      colors = [red_color if idx % 2 == 0 else firoza_color for idx, _ in enumerate(class_names)]

      h, w, _ = img.shape

      for r in result_seg:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

      if masks is not None:
          masks = masks.data.cpu()
          for seg, box in zip(masks.data.cpu().numpy(), boxes):

              seg = cv2.resize(seg, (w, h))
              img = overlay(img, seg, colors[int(box.cls)], 0.4)

              xmin = int(box.data[0][0])
              ymin = int(box.data[0][1])
              xmax = int(box.data[0][2])
              ymax = int(box.data[0][3])

              plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{class_names[int(box.cls)]} {float(box.conf):.3}')


      # Work with od (box over cattle body) processing
      x1 = bounding_box[0][0]
      y1 = bounding_box[0][1]
      x2 = bounding_box[0][2]
      y2 = bounding_box[0][3]

      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

      # Calculate the middle points of the rectangle
      middle_x = (x1 + x2) // 2
      middle_y = (y1 + y2) // 2

      cv2.line(img, (x1, middle_y), (x2, middle_y), (0, 0, 255), 2)

      cv2.line(img, (middle_x, y1), (middle_x, y2), (0, 0, 255), 2)

      cv2.putText(img, f"Estimated BODY LENGTH: {inch_b_w} inch", (middle_x, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
      cv2.putText(img, f"Estimated HEART GIRTH: {inch_b_h} inch", (middle_x, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

      ig = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

      return ig

"""----------------------------------------------------------------"""



""" Measurement """
"""---------------------------------------------------------------"""

def measurement(ref_w, ref_h, bbox_w, bbox_h, scale = None):            # calculation of pixel evqulent to inch, reference object size set to 4 inch.

  if scale is None:
    mean = (ref_w + ref_h)/2
    scale = round(mean / (4*2))

  inch_b_w = round(bbox_w / scale)
  inch_b_h = round(bbox_h / scale)
  weight = round(((inch_b_w*(inch_b_h**2))/300)/2)

  return inch_b_w, inch_b_h, weight




"""Main process"""

""" ---------------------------------------------------------------"""



def mainloop(image, result_seg, result_od):

  for r in result_od:
    bbox_w = round((r.boxes.xywh).numpy()[0][2])     # ob bounding box width
    bbox_h = round((r.boxes.xywh).numpy()[0][3])     # ob bounding box height
    od_b_box = (r.boxes.xyxy).numpy()                # ob bounding box xy coordination


  for r in result_seg:                              #seg
    boxes = r.boxes.data
    clss = (boxes[:, 5]).numpy()

  if 0. in clss and 1.0 in clss:
    ref_w = (r.boxes[1].xywh).numpy()[0][2]   # reference segment box width
    ref_h = (r.boxes[1].xywh).numpy()[0][3]   # reference segment box height

    if ref_w > 200 or ref_h > 200 :
      ref_w = (r.boxes[0].xywh).numpy()[0][2]
      ref_h = (r.boxes[0].xywh).numpy()[0][3]

    inch_b_w, inch_b_h, weight = measurement(ref_w, ref_h, bbox_w, bbox_h)
    get_img = image_pro(image, result_seg, od_b_box, inch_b_w, inch_b_h)

    return weight, get_img, 2


  elif 0. in clss:
    ref_w, ref_h = 0, 0
    scl = 13
    inch_b_w, inch_b_h, weight = measurement(ref_w, ref_h, bbox_w, bbox_h, scl)
    get_img = image_pro(image, result_seg, od_b_box, inch_b_w, inch_b_h)

    return weight, get_img, 1



  else:

    return 0


