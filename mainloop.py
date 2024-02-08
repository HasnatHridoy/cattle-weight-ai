
import cv2 as cv2



def mainloop(file_n, result_seg, result_od):

  def measurement(ref_w, ref_h, bbox_w, bbox_h):            # calculation of pixel evqulent to inch, reference object size set to 4 inch.

    mean = (ref_w + ref_h)/2
    scl = round(mean / (4*2))  

    inch_b_w = round(bbox_w / scl)
    inch_b_h = round(bbox_h / scl)
    weight = round(((inch_b_w*(inch_b_h**2))/300)/2)

    return inch_b_w, inch_b_h, weight



  def img_show(file_name, bounding_box, inch_b_w, inch_b_h):              # process image to show

    image_loc ='/content/runs/segment/predict/'+file_name
    image = cv2.imread(image_loc)

    x1 = bounding_box[0][0]
    y1 = bounding_box[0][1]
    x2 = bounding_box[0][2]
    y2 = bounding_box[0][3]

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Calculate the middle points of the rectangle
    middle_x = (x1 + x2) // 2
    middle_y = (y1 + y2) // 2

    cv2.line(image, (x1, middle_y), (x2, middle_y), (0, 0, 255), 2)

    cv2.line(image, (middle_x, y1), (middle_x, y2), (0, 0, 255), 2)

    cv2.putText(image, f"Estimated size: {inch_b_w} inch", (x2, middle_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Estimated size: {inch_b_h} inch", (middle_x, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    img = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img



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

    get_img = img_show(file_n, od_b_box, inch_b_w, inch_b_h)

    return inch_b_w, inch_b_h, weight, get_img

  elif 0. in clss:
    print('No reference object detected, measurement might be incorrect.')

  else:
    print("nothing here")