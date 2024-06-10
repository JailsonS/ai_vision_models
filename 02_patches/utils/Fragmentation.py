import numpy as np
from skimage.measure import label
from scipy.ndimage import find_objects
from collections import defaultdict

class ClassifyPatches:

    obj_id = 1

    def __init__(self, stack_binary: list) -> None:
        
        # map object id to its parent
        self.parent_map = {}  
        
        # map object id to its slice
        self.obj_map = {}  

        # list to hold the classified arrays
        self.history = []  
        
        self.stack_binary = stack_binary

        # list to hold the result arrays
        self.result_arrays = []

    def get_unique_id(self):
        unique_id = ClassifyPatches.obj_id
        ClassifyPatches.obj_id += 1
        return unique_id



    def update_object_map(self, labeled_array, current_objects):
        objects = find_objects(labeled_array)
        obj_dict = {}
        for obj_idx, obj_slice in enumerate(objects, start=1):
            if obj_idx in current_objects:
                obj_id = current_objects[obj_idx]
            else:
                obj_id = self.get_unique_id()
            obj_dict[obj_idx] = obj_id
            self.obj_map[obj_id] = obj_slice
        return obj_dict



    def find_connected_objects(self, prev_array, current_array):
        connections = defaultdict(set)
        for prev_label in np.unique(prev_array):

            if prev_label == 0: continue

            prev_mask = prev_array == prev_label
            overlap_labels = current_array[prev_mask]
            overlap_labels = overlap_labels[overlap_labels != 0]
            unique_labels = np.unique(overlap_labels)
            
            for label in unique_labels:
                connections[label].add(prev_label)
        
        return connections
    

    
    def classify(self):
        for t, array in enumerate(self.stack_binary):

            

            proj = {
                'crs':array.crs,
                'transform':array.transform
            }

            array = array.read()[0]

            labeled_array, num_features = label(array, return_num=True, connectivity=1)
            current_objects = self.update_object_map(labeled_array, {})
            if t == 0:
                parent_map = {obj_id: obj_id for obj_id in current_objects.values()}
            else:
                prev_labeled_array, _ = self.history[-1]
                connections = self.find_connected_objects(prev_labeled_array, labeled_array)
                for curr_label, prev_labels in connections.items():
                    if len(prev_labels) == 1:
                        prev_label = list(prev_labels)[0]
                        current_objects[curr_label] = prev_label
                    else:
                        current_objects[curr_label] = self.get_unique_id()
                current_objects = self.update_object_map(labeled_array, current_objects)

            self.history.append((labeled_array, current_objects))
            result_array = np.zeros_like(labeled_array)
            
            for obj_idx, obj_id in current_objects.items():
                result_array[labeled_array == obj_idx] = obj_id
            
            self.result_arrays.append([result_array, proj])
        
        return self.result_arrays, parent_map, self.history
