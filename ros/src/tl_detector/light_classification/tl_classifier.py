from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import cv2
import tensorflow as tf
import os 


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        
        self.current_light = TrafficLight.UNKNOWN
        self.category_index = {1: 'Green', 2: 'Red', 3: 'Yellow', 4: 'Off'}
        #self.category_index = {1: 'Off', 2: 'Green', 3: 'Yellow', 4: 'Red'}  # Bosch 
        
        #relative_path = '/transfer_ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
        #relative_path = '/model2/frozen_inference_graph.pb'
        relative_path = '/models/ssd_sim/frozen_inference_graph.pb'
        cwd = os.path.dirname(os.path.realpath(__file__))
        model_path = cwd + relative_path
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
    
            
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
    
        config = tf.ConfigProto()
        self.sess = tf.Session(graph=self.graph, config=config)
 
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        self.current_light = TrafficLight.UNKNOWN
        
        #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(image, axis=0)

        with self.graph.as_default():
            boxes, scores, classes, num = self.sess.run([self.detection_boxes, self.detection_scores,
                                                      self.detection_classes, self.num_detections],
                                                      feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        confidence_cutoff = 0.5
        red_ratio_th = 0.1
       
        count_total = 0
        count_red = 0
        count_green = 0
        
        for i in range(len(boxes)):
           if scores is None or scores[i] > confidence_cutoff: 
                class_clr = self.category_index[classes[i]]
                count_total += 1.0
                if class_clr == "Red":
                    count_red += 1
                elif class_clr == "Green": 
                    count_green += 1
            
        if count_red / (count_total+0.00001) > red_ratio_th:
            self.current_light = TrafficLight.RED
        #elif count_green > 0 and count_green > count_red:
        #    self.current_light = TrafficLight.GREEN
        else: self.current_light = TrafficLight.GREEN
        
        return self.current_light
