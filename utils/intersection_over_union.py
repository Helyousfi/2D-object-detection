def intersection_over_union(boxes_preds, boxes_labels):
    
    # boxes_preds[0] : upper left X
    # boxes_preds[1] : upper left y
    # boxes_preds[2] : width
    # boxes_preds[3] : height
    
    # for predicted bbox 
    box1_x1 = boxes_preds[0]
    box1_y1 = boxes_preds[1]
    box1_x2 = boxes_preds[0] + boxes_preds[2]
    box1_y2 = boxes_preds[1] + boxes_preds[3]  
    # For targeted bbox
    box2_x1 = boxes_labels[0]
    box2_y1 = boxes_labels[1]
    box2_x2 = boxes_labels[0] + boxes_labels[2]
    box2_y2 = boxes_labels[1] + boxes_labels[3]

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    if x1 > x2 and y1 > y2:
        intersection = 0
    else:
        intersection = abs(x2 - x1) * abs(y2 - y1)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

if __name__ == "__main__": 
    bbox_label = [0, 0, 100, 100]
    bbox_target = [10, 10, 20, 20]

    iou = intersection_over_union(bbox_target, bbox_label)
    print(iou)
