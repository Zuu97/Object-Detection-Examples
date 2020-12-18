import cv2 as cv
from matplotlib import pyplot as plt 

plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')

def yxyx_to_xyxy(box):
    ymin, xmin, ymax, xmax = box
    return map(int, [xmin, ymin, xmax, ymax])

def xyxy_to_yxyx(box):
    xmin, ymin, xmax, ymax = box
    return map(int, [ymin, xmin, ymax, xmax])

def visualize_bbox(image, box, color=color, thickness=thickness):
    ymin, xmin, ymax, xmax = box

    image_width = image.shape[1]
    image_height = image.shape[0]

    assert (int(ymax) - int(ymin) > image_width), " invalid image height"
    assert (int(xmax) - int(xmin) > image_height), " invalid image width"

    cv.rectangle(
            image, 
            (int(xmin), int(ymin)), (int(xmax), int(ymax)), 
            color, 
            thickness
            )

def visualize_bboxes(image, boxes, color=[], thickness=thickness):
    assert (len(boxes.shape) == 2) and (boxes.shape[1] == 4), "Invalid bounding box shapes"

    for i in range(boxes.shape[0]):
        box = boxes[i]
        visualize_bbox(image, box, color[i], thickness)
    return image

def scale_boxes(bbox, image, bboxes_normalized = True):
    xmin, ymin, xmax, ymax = bbox
    image_width = image.shape[1]
    image_height = image.shape[0]

    if bboxes_normalized == True:
        bbox = [
              xmin * image_width, 
              ymin * image_height, 
              xmax * image_width, 
              ymax * image_height
              ]
    return bbox

def display_digits_with_boxes(images, pred_bboxes, bboxes, iou, title, bboxes_normalized=False):

    n = len(images)

    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])
  
    for i in range(n):
      ax = fig.add_subplot(1, 10, i+1)
      bboxes_to_plot = []
      image = images[i]
      if (len(pred_bboxes) > i):
        bbox = pred_bboxes[i]
        bbox = scale_boxes(bbox, image, True)
        bboxes_to_plot.append(bbox)
    
      if (len(bboxes) > i):
        bbox = bboxes[i]
        bbox = scale_boxes(bbox, image, bboxes_normalized)
        bboxes_to_plot.append(bbox)

      img_to_draw = visualize_bboxes(
                              image=image, 
                              boxes=np.asarray(bboxes_to_plot), 
                              color=[(255,0,0), (0, 255, 0)]
                              )
      plt.xticks([])
      plt.yticks([])
    
      plt.imshow(img_to_draw)

      if len(iou) > i :
        color = "black"
        if (iou[i][0] < iou_threshold):
          color = "red"
        ax.text(0.2, -0.3, "iou: %s" %(iou[i][0]), color=color, transform=ax.transAxes)
        
def plot_metrics(history, metric_name, title, ylim=ylim):

    train_metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(train_metric, color='blue', label=metric_name)
    plt.plot(val_metric, color='green', label='val_' + metric_name)

def standard_normalize_images(image):
    image = image/127.5
    image = image - 1
    return image

def read_tensorflow_images(image, bbox):
    image = tf.cast(image, tf.float32)
    image_width = tf.shape(image)[1]
    image_height = tf.shape(image)[0]

    norm_x = tf.cast(
                    image_width, 
                    tf.float32
                    )
    norm_y = tf.cast(
                    image_height, 
                    tf.float32
                    )

    image = tf.image.resize(
                        image, 
                        image_size
                        )

    image = standard_normalize_images(image)

    xmin, ymin, xmax, ymax = bbox

    bbox_arr = [xmin / norm_x , 
                 ymin / norm_y, 
                 xmax / norm_x , 
                 ymax / norm_y]
    
    return image, bbox_arr

def read_tensorflow_image_with_original(image, bbox):

    original_img = image
    image, bbox_arr = read_tensorflow_images(image, bbox)