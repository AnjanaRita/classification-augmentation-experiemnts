from .utility import get_line_bbox, get_poly_format, calculate_iou, get_width_height, partial_text_matcher
import re
import attr
import random
import pandas as pd
from fuzzywuzzy import fuzz

@attr.s
class AddresslineTagger:
    """
    Converts the Raw documents into weakly supervised training dataset for addressline vs non-addressline classification.
    """
    dataset = attr.ib()
    threshold = attr.ib(default=0.4)
    
    @property
    def result(self):
        result_df = []
        columns = ['document_id', 'line_id', 'text','word_bbox', 'label', 'class_2_target', 'class_3_target']
        for _doc in self.dataset:
            lines = _doc['document']
            ground_truth = _doc['ground_truth']

            for i in lines:
                i['2_class'] = 'non-addressline'
                i['3_class'] = 'non-addressline'
                i['label'] = []

            line_text = [i['text'] for i in lines]
            for label, value in ground_truth.items():
                line_index = partial_text_matcher(value, line_text )
                lines = self.assign_line_type(value['text'], lines,line_index, label)
        
            thresold = int(len(lines)*self.threshold)
            df = [(_doc['doc_id'],i['idx'],i['text'], i['word_bbox'], i['label'], i['2_class'],i['3_class'],) for i in lines[:]]
            df = pd.DataFrame(df, columns=columns)
            negative_sample = df.query('class_2_target == "non-addressline"')
            positive_sample = df.query('class_2_target != "non-addressline"')
            if thresold > len(negative_sample):
                thresold = len(negative_sample)
            negative_sample = negative_sample.sample(n=thresold)
            df_1 = pd.concat([negative_sample, positive_sample]).sort_values(by=['line_id'])
            result_df.append(df_1)
        return pd.concat(result_df)
    
    def assign_line_type(self, text, lines, line_index, label):
        for i in line_index:
            lines[i]['2_class'] = 'addressline'
            lines[i]['label'].append(label)

        if len(line_index) == 1:
            lines[line_index[0]]['3_class'] = 'full-addressline'
            if label not in lines[line_index[0]]['label']:
                lines[line_index[0]]['label'].append(label)
            return lines

        text = text.replace('\n',' ')
        for index in line_index:
            line_text = lines[index]['text']

            if label not in lines[index]['label']:
                lines[index]['label'].append(label)

            if self._is_good_line(line_text) and fuzz.partial_ratio(text, line_text) > 80:
                lines[index]['3_class'] = 'full-addressline' 
            else:
                lines[index]['3_class'] = 'partial-addressline' 
        return lines
    
    def _is_good_line(self,text):
        text = re.sub('\d+','', text)
        text = re.sub('  +',' ', text)
        text = text.split(' ')
        return len(text) > 1


@attr.s
class BoxBasedTagger:
    """
    Creats a noisy/weak supervised dataset for buyer vs vendor classification.
    """
    dataset = attr.ib()
    thresold = attr.ib(default=1)
    
    @property
    def result(self):
        data = [self.data_tagger(i) for i in self.dataset]
        return pd.concat(data)
    
    
    def get_label(self, labels):
        return_label = 'non-addressline'
        if len(labels) == 1:
            return_label = labels[0].replace('_','-')
        elif len(labels) == 2:
            return_label = 'both'
        return return_label

    def data_tagger(self, sample):
        tagged_data = []
        lines = sample['document']
        ground_truth = sample['ground_truth']
        thresold = int(len(lines)* self.thresold)
        width, height = get_width_height(lines)
        for line in lines[:thresold]:
            line_bbox = {}
            label_1 = []
            word_bbox = line['word_bbox']
            if len(word_bbox) == 0:
                continue
            line_bbox = get_line_bbox(word_bbox)
            for k, v in ground_truth.items():
                actual_data = v['region']
                area = calculate_iou(line_bbox, v['region'])
                if area > 0.0:
                    label_1.append(k)
            label = self.get_label(label_1)
            _ = dict(text=line['text'], x1=line_bbox['x1']/width, 
                     y1=line_bbox['y1']/height,  x2=line_bbox['x2']/width,
                     y2=line_bbox['y2']/height,target = label, line_id=line['idx'] ,doc_idx = sample['doc_id'])
            if label == 'non-addressline':
                if random.randint(0,15)% 3 == 0:
                    tagged_data.append(_)
            else:
                tagged_data.append(_)
        return pd.DataFrame(tagged_data)