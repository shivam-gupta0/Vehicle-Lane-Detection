import cv2
import numpy as np


cfg_path = "E:\\project\\car_yolov3\\yolo_pretrained\\yolov3.cfg"
weights_path = "E:\\project\\car_yolov3\\yolo_pretrained\\yolov3-spp.weights"


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('E:\master_thesis\project\vehicle_dataset\model_test\\output.mp4', fourcc, 20.0, (640, 480))


def detection(image, cfg_path, weights_path):
    ht, wt, _ = image.shape
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    last_layer = net.getUnconnectedOutLayersNames()
    layer_out = net.forward(last_layer)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_out:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > .6:
                center_x = int(detection[0] * wt)
                center_y = int(detection[1] * ht)
                w = int(detection[2] * wt)
                h = int(detection[3] * ht)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


classes = []
with open("E:\\project\\car_yolov3\\yolo_pretrained\\coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
cap = cv2.VideoCapture("E:\\project\\vehicle_dataset\\model_test\\3.webm")
while cap.isOpened():
    rate, frame = cap.read()
    img = frame
    l1 = cv2.imread("E:\\project\\vehicle_dataset\\model_test\\mask_vid1\\44_lane1.jpg")
    l2 = cv2.imread("E:\\project\\vehicle_dataset\\model_test\\mask_vid1\\44_lane2.jpg")
    l3 = cv2.imread("E:\\project\\vehicle_dataset\\model_test\\mask_vid1\\44_lane3.jpg")
    l4 = cv2.imread("E:\\project\\vehicle_dataset\\model_test\\mask_vid1\\44_lane4.jpg")
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    my_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    ml1 = cv2.bitwise_and(my_img, cv2.resize(l1, (width, height), interpolation=cv2.INTER_AREA))
    ml2 = cv2.bitwise_and(my_img, cv2.resize(l2, (width, height), interpolation=cv2.INTER_AREA))
    ml3 = cv2.bitwise_and(my_img, cv2.resize(l3, (width, height), interpolation=cv2.INTER_AREA))
    ml4 = cv2.bitwise_and(my_img, cv2.resize(l4, (width, height), interpolation=cv2.INTER_AREA))

    boxes1, confidences1, class_ids1 = detection(ml1, cfg_path, weights_path)
    boxes2, confidences2, class_ids2 = detection(ml2, cfg_path, weights_path)
    boxes3, confidences3, class_ids3 = detection(ml3, cfg_path, weights_path)
    boxes4, confidences4, class_ids4 = detection(ml4, cfg_path, weights_path)

    indexes1 = cv2.dnn.NMSBoxes(boxes1, confidences1, .5, .4)
    indexes2 = cv2.dnn.NMSBoxes(boxes2, confidences2, .5, .4)
    indexes3 = cv2.dnn.NMSBoxes(boxes3, confidences3, .5, .4)
    indexes4 = cv2.dnn.NMSBoxes(boxes4, confidences4, .5, .4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors1 = np.random.uniform(0, 255, size=(len(boxes1), 3))
    colors2 = np.random.uniform(0, 255, size=(len(boxes2), 3))
    colors3 = np.random.uniform(0, 255, size=(len(boxes3), 3))
    colors4 = np.random.uniform(0, 255, size=(len(boxes4), 3))

    if len(indexes1) > 0:
        for i in indexes1.flatten():
            x, y, w, h = boxes1[i]

            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)

            label = str(classes[class_ids1[i]])
            confidence = str(round(confidences1[i], 2))
            color = colors1[i]
            cv2.rectangle(my_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(my_img, "lane1", (x, y + 20), font, 3, (255, 255, 0), 1)
    if len(indexes2) > 0:
        for i in indexes2.flatten():
            x, y, w, h = boxes2[i]

            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)

            label = str(classes[class_ids2[i]])
            confidence = str(round(confidences2[i], 2))
            color = colors2[i]
            cv2.rectangle(my_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(my_img, "lane2", (x, y + 20), font, 3, (255, 255, 0), 1)
    if len(indexes3) > 0:
        for i in indexes3.flatten():
            x, y, w, h = boxes3[i]

            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)

            label = str(classes[class_ids3[i]])
            confidence = str(round(confidences3[i], 2))
            color = colors3[i]
            cv2.rectangle(my_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(my_img, "lane3", (x, y + 20), font, 3, (255, 255, 0), 1)
    if len(indexes4) > 0:
        for i in indexes4.flatten():
            x, y, w, h = boxes4[i]

            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)

            label = str(classes[class_ids4[i]])
            confidence = str(round(confidences4[i], 2))
            color = colors4[i]
            cv2.rectangle(my_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(my_img, "lane4", (x, y + 20), font, 3, (255, 255, 0), 1)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('image', 1200, 600)
    cv2.imshow('image', my_img)

    if cv2.waitKey(1) == ord('q'):
        continue
    elif cv2.waitKey(1) == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
