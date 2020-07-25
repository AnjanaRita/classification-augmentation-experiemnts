import os
import json
import random
import pickle
import jsonlines
from pprint import pprint
from fuzzywuzzy import fuzz
from sklearn.metrics import *
from shapely.geometry import Polygon
from sklearn.metrics import accuracy_score

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model,f)
        

def pickle_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def jsonl_reader(file_path):
    dataset = []
    if not os.path.isfile(file_path):
        raise FileNotFoundError('Please provide the proper file path')
    with jsonlines.open(file_path) as reader:
        dataset = [i for i in reader]
    return dataset


def inspection_full_matching(document, n=0):
    ground_truth = document['ground_truth']
    line_data = [i['text'] for i in document['document']]
    pprint(ground_truth)
    print()
    pprint(line_data[n:n+5])
    return

def print_classifaction_report(model, data, actual_data):
    print(confusion_matrix(actual_data, model.predict(data)))
    print(classification_report(actual_data, model.predict(data)))
    return

def inspection_partial_matching(document):
    line_data = [i['text']  for i in document['document']]
    ground_text = document['ground_truth']
    for i,j in ground_text.items():
        line_index = partial_text_matcher(j,line_data)
        match_string  = '\n'.join([line_data[i] for i in line_index])
        print(f"{i}\nGround Truth: {j['text']}\n")
        print(f"Actual Present Text: {match_string}")
        print('*'*10)
    return

def bbox_formating(bbox):
    return_list = []
    for i in bbox:
        temp = dict(x1=i['x'], y1=i['y'], x2=i['x']+i['width'], y2=i['y']+i['height'])
        return_list.append(temp)
    return return_list
        
        
def get_line_bbox( bbox):
    bbox = bbox_formating(bbox)
    x1 = min([i['x1'] for i in bbox])
    y1 = min([i['y1'] for i in bbox])
    x2 = max([i['x2'] for i in bbox])
    y2 = max([i['y2'] for i in bbox])
    return dict(x1=x1, y1=y1, x2=x2, y2=y2)

def get_poly_format(data):
    x1 , y1 = data['x1'], data['y1']
    x2 , y2 = data['x2'], data['y2']
    return [(x1,y1),(x1, y2), (x2, y1), (x2, y1)]

def calculate_iou(box_1, box_2):
    try:
        box_1 = get_poly_format(box_1)
        box_2 = get_poly_format(box_2)
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        
    except Exception as ex:
        print(box_1, box_2)
        iou = 0.0
    return iou

def get_width_height(lines):
    boxes = [get_line_bbox(i['word_bbox']) for i in lines if len(i['word_bbox'])]
    width = max([i['x2'] for i in boxes]) + random.randint(40, 80)*2
    height = max([i['y1'] for i in boxes]) + random.randint(40, 80)*2
    return width, height

def full_text_matcher(ground_truth, list_data):
    text = ground_truth['text']
    return any([i for i in list_data if text in i])

def partial_text_matcher(ground_truth,list_data ):
    line_index = []
    
    _text = ground_truth['text']
    if full_text_matcher(ground_truth, list_data):
        line_index = [ind for ind,i in enumerate(list_data) if _text in i]
        return sorted(line_index)
    

    for text in _text.split('\n'):
        for index, ref in enumerate(list_data):
            if len(ref) < 5:
                continue
            score_1 = fuzz.partial_ratio(text, ref)
            score_2 = fuzz.partial_token_set_ratio(text, ref)
            if (score_1 >=60 and score_2 > 80) and index not in line_index:                        
                line_index.append(index)
                
    return sorted(line_index)

def formating(data):
    data['label'] = None
    data['line_id'] = None
    data['word_bbox'] = None
    data['document_id'] = None
    return data