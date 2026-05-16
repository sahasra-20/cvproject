import numpy as np
from sklearn.cluster import DBSCAN
import sys
from pathlib import Path

# Add project root to path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from configs.config import THRESHOLDS

class RiderBikeAssociator:
    def __init__(self):
        self.fallback_iou = THRESHOLDS.get("rider_bike_iou_fallback", 0.15)
        
    def _is_point_in_box(self, pt, box):
        x, y = pt
        x1, y1, x2, y2 = box
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    def _calculate_iou(self, boxA, boxB):
        # Calculate Intersection over Union
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea)

    def _get_centroid(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def associate(self, bike_boxes, rider_boxes, rider_keypoints):
        """
        Maps bounding boxes of persons (riders) to the bounding boxes of motorcycles.
        Returns a list of dictionaries mapping 'bike' to its respective 'riders'.
        """
        associations = [{'bike': b, 'riders': [], 'rider_kpts': []} for b in bike_boxes]
        unassigned_riders = []

        # PRIMARY LOGIC: Keypoint Geometric Proximity
        # Check if the Rider's Hip or Knee (COCO indices 11,12,13,14) physically sits inside the bike's bounding box
        for r_idx, (r_box, r_kpts) in enumerate(zip(rider_boxes, rider_keypoints)):
            assigned = False
            hip_knee_indices = [11, 12, 13, 14]
            valid_pts = []
            
            if len(r_kpts) > 0:
                for idx in hip_knee_indices:
                    if idx < len(r_kpts) and r_kpts[idx][2] > 0: # conf > 0
                        valid_pts.append((r_kpts[idx][0], r_kpts[idx][1]))
            
            best_bike_idx = -1
            best_pts_in = 0
            
            for b_idx, b_box in enumerate(bike_boxes):
                pts_in = sum(1 for pt in valid_pts if self._is_point_in_box(pt, b_box))
                if pts_in > best_pts_in:
                    best_pts_in = pts_in
                    best_bike_idx = b_idx
            
            if best_bike_idx != -1:
                associations[best_bike_idx]['riders'].append(r_box)
                associations[best_bike_idx]['rider_kpts'].append(r_kpts)
                assigned = True
            
            #FALLBACK LOGIC: IoU & Centroid height
            # If hips are occluded, check if IoU > 15% AND the person is physically positioned above the bike
            if not assigned:
                r_cent = self._get_centroid(r_box)
                best_iou = 0
                
                for b_idx, b_box in enumerate(bike_boxes):
                    b_cent = self._get_centroid(b_box)
                    iou = self._calculate_iou(r_box, b_box)
                    
                    if iou > self.fallback_iou and r_cent[1] < b_cent[1] and iou > best_iou:
                        best_iou = iou
                        best_bike_idx = b_idx
                        
                if best_bike_idx != -1:
                    associations[best_bike_idx]['riders'].append(r_box)
                    associations[best_bike_idx]['rider_kpts'].append(r_kpts)
                    assigned = True
                    
            if not assigned:
                unassigned_riders.append((r_idx, r_box, r_kpts))
                
        # DENSE TRAFFIC FALLBACK: DBSCAN Clustering
        # If no match riders Stored separately and then cluster them based on their centroids
        # In highly crowded scenes where riders and bikes heavily overlap
        if len(unassigned_riders) > 0 and len(bike_boxes) > 0:
            unassigned_cents = np.array([self._get_centroid(r[1]) for r in unassigned_riders])
            bike_cents = np.array([self._get_centroid(b) for b in bike_boxes])
            
            # eps=60: max pixel distance to cluster riders
            clustering = DBSCAN(eps=60, min_samples=1).fit(unassigned_cents)
            
            for label in set(clustering.labels_):
                if label == -1: continue
                
                cluster_pts = unassigned_cents[clustering.labels_ == label]
                cluster_cent = np.mean(cluster_pts, axis=0)
                
                # Calculate vector distance to all bike centroids
                dists = np.linalg.norm(bike_cents - cluster_cent, axis=1)
                nearest_bike_idx = int(np.argmin(dists))
                
                # Bulk assign the clustered riders
                for idx, r_item in enumerate(unassigned_riders):
                    if clustering.labels_[idx] == label:
                        _, r_box, r_kpts = r_item
                        associations[nearest_bike_idx]['riders'].append(r_box)
                        associations[nearest_bike_idx]['rider_kpts'].append(r_kpts)

        # FINAL INFERENCE: Guarantee minimum 1 rider
        for assoc in associations:
            if len(assoc['riders']) == 0:
                assoc['inferred_rider'] = True # Assumed ghost rider for violation triggers
            else:
                assoc['inferred_rider'] = False

        return associations

