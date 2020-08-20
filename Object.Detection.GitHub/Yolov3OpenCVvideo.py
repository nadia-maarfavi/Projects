import cv2
import numpy as np
import time
import os
import glob
import pandas as pd

net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") # Original yolov3
#net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

vid_dir = "Videos" # Enter Directory of all images
data_path = os.path.join(vid_dir,'*.mp4')
files = glob.glob(data_path)
data = []

#create empty dataframe
ObjectsDataframe = pd.DataFrame(
    columns=['x-cordinate', 'y-cordinate', 'w-cordinate', 'h-cordinate', 'Confidence',
             'c_id1', 'c_id2', 'c_id3', 'c_id4', 'c_id5', 'c_id6','c_id7', 'c_id8', 'c_id9',
             'c_id10','c_id11', 'c_id12', 'c_id13', 'c_id14', 'c_id15', 'c_id16', 'c_id17',
             'c_id18', 'c_id19','c_id20', 'c_id21', 'c_id22', 'c_id23', 'c_id24', 'c_id25', 'c_id26', 'c_id27',
             'c_id28','c_id29', 'c_id30', 'c_id31','c_id32', 'c_id33', 'c_id34', 'c_id35', 'c_id36', 'c_id37', 'c_id38',
             'c_id39','c_id40', 'c_id41', 'c_id42', 'c_id43',
             'c_id44', 'c_id45', 'c_id46', 'c_id47', 'c_id48', 'c_id49', 'c_id50','c_id51',
             'c_id52', 'c_id53', 'c_id54', 'c_id55', 'c_id56', 'c_id57',
             'c_id58', 'c_id59', 'c_id60', 'c_id61', 'c_id62', 'c_id63', 'c_id64',
             'c_id65','c_id66', 'c_id67', 'c_id68', 'c_id69', 'c_id70',
             'c_id71', 'c_id72', 'c_id73', 'c_id74', 'c_id75', 'c_id76', 'c_id77','c_id78',
             'c_id79', 'c_id80','Video.Name','Frame.Number'])
print("ObjectDataframeInfo", ObjectsDataframe.info())

for f1 in files:
    print("File are: ", f1)
    VideoName = str(f1)

    # loading image
    cap = cv2.VideoCapture(f1)  # 0 for 1st webcam
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0

    dictionclasses = {}
    objects = []
    while (cap.isOpened()):
                ret,frame = cap.read()  #
                if ret == False:
                    print("Video Finished !!!")
                    break
                frame_id += 1
                print("Frame ID: ",frame_id)
                height, width, channels = frame.shape
                # detecting objects
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # reduce 416 to 320

                net.setInput(blob)
                outs = net.forward(outputlayers)
                # print(outs[1])

                # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
                class_ids = []
                confidences = []
                boxes = []
                objectsvector = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.3:
                            # onject detected
                            objectsvector.append(detection)
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                            # rectangle co-ordinaters
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                            boxes.append([x, y, w, h])  # put all rectangle areas
                            confidences.append(float(confidence))  # how confidence was that object detected and show that percentage
                            class_ids.append(class_id)  # name of the object tha was detected

                objects = []
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

                for i in range(len(objectsvector)):
                    if i in indexes:
                        objects.append(objectsvector[i])

                objects = np.array(objects)
                print("objects.shape: ", objects.shape)
                print(type(objects))

                df2 = pd.DataFrame(objects,
                                   columns=['x-cordinate', 'y-cordinate', 'w-cordinate', 'h-cordinate', 'Confidence',
                                            'c_id1', 'c_id2', 'c_id3', 'c_id4', 'c_id5', 'c_id6', 'c_id7', 'c_id8',
                                            'c_id9','c_id10', 'c_id11', 'c_id12', 'c_id13', 'c_id14', 'c_id15', 'c_id16',
                                            'c_id17','c_id18', 'c_id19', 'c_id20', 'c_id21', 'c_id22', 'c_id23', 'c_id24',
                                            'c_id25', 'c_id26', 'c_id27','c_id28', 'c_id29', 'c_id30', 'c_id31', 'c_id32', 'c_id33', 'c_id34',
                                            'c_id35', 'c_id36', 'c_id37', 'c_id38','c_id39', 'c_id40', 'c_id41', 'c_id42', 'c_id43',
                                            'c_id44', 'c_id45', 'c_id46', 'c_id47', 'c_id48', 'c_id49', 'c_id50',
                                            'c_id51','c_id52', 'c_id53', 'c_id54', 'c_id55', 'c_id56', 'c_id57',
                                            'c_id58', 'c_id59', 'c_id60', 'c_id61', 'c_id62', 'c_id63', 'c_id64',
                                            'c_id65', 'c_id66', 'c_id67', 'c_id68', 'c_id69', 'c_id70',
                                            'c_id71', 'c_id72', 'c_id73', 'c_id74', 'c_id75', 'c_id76', 'c_id77',
                                            'c_id78','c_id79', 'c_id80'])

                df2['Video.Name'] = VideoName
                df2['Frame.Number'] = frame_id
                ObjectsDataframe = ObjectsDataframe.append(df2)

                ObjectsDataframe.to_csv(r'SampleOutput.csv', index=False)

                elapsed_time = time.time() - starting_time
                fps = frame_id / elapsed_time
                cv2.putText(frame, "FPS:" + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 1)

                print("Time per frame: ",fps)
    # key = cv2.waitKey(1)  # wait 1ms the loop will start again and we will process the next frame



    cap.release()
#cv2.destroyAllWindows()