import attr
import pickle
import pandas as pd
from .utility import get_width_height, pickle_load, get_line_bbox
from .features import FeatureGeneration


addresslines = ['avg_token_length', 'more_then_avg', 'alpha_count',
               'stop_word_count', 'punct_count', 'digit_count', 'alpha_num_count',
               'number_of_title_tokens', 'not_alphabet_count', 'number_of_tokens',
               'dolar_count','address_key_words','noun_count','upper_case']
buyer_vs_supplier = ['x1','y1','x2','y2', 'line_id']

@attr.s
class Inference:
    """
    Infernece utility for new documents.
    """
    data = attr.ib()
    thresold = attr.ib(repr=False, init=False)
    
    def __attrs_post_init__(self):
        self.thresold = int(len(self.data)*0.40)
        
    @property
    def predict(self):
        data = self.feature_generation()
        model_1 = pickle_load('models/svm_aug.pkl')
        model_2 = pickle_load('models/rfc_address_classifier.pkl')
        data['line_classifier'] = model_1.predict(data[addresslines])
        data['address_classifier']= model_2.predict(data[buyer_vs_supplier])
        data = self.output_formating(data)
        return data[['line_id','text', 'line_classifier','address_classifier']]
        
    def output_formating(self, data):
        data = data[['text','line_id', 'line_classifier','address_classifier']]
        output = [(i['idx'], i['text']) for i in self.data]
        dataframe = pd.DataFrame(output[self.thresold:], columns=['line_id', 'text'])
        dataframe['line_classifier'] = 'non-addressline'
        dataframe['address_classifier'] = 'non-addressline'
        return pd.concat([data, dataframe])
    
    def feature_generation(self):
        df1 = FeatureGeneration(pd.DataFrame(self.data[:self.thresold], columns=['text'])).transform
        df2 = self._formating()
        return df2.merge(df1, on='text', how='right')
        
        
    def _formating(self):
        dataframe = []
        width, height = get_width_height(self.data)
        for line in self.data[:self.thresold]:
            word_bbox = line['word_bbox']
            if len(word_bbox) == 0:
                continue
            lb = get_line_bbox(word_bbox)
            temp = dict(line_id = line['idx'], text = line['text'],
                        x1 = lb['x1']/width, x2 = lb['x2']/width,
                        y1 = lb['y1']/height, y2 = lb['y2']/height)
            dataframe.append(temp)
        return pd.DataFrame(dataframe)