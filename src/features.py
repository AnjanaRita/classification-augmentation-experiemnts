import re
import attr
import spacy
import numpy as np
import pandas as pd

nlp = spacy.load('en_core_web_sm')

can_address_words = ['rd ','rd,' ,'rd.','road', 'street','st ', 'st.', 'st'
                     'way','plaza','market','blvd','floor','flr',
                     'avenue','building', 'park',
                     'block','blk','place','tower', 'ave',]


@attr.s
class FeatureGeneration:
    """
    Utility class to calculate various features for a given document.
    """
    dataframe = attr.ib()
    
    @property
    def transform(self):
        feature_dataset = []
        for row in self.dataframe.itertuples():
            result = self.get_feature(row.text)
            feature_dataset.append(result)
        return pd.DataFrame.from_dict(feature_dataset, orient='columns')
        
    def _avg_token_lengh(self, text):
        return np.mean([len(i.text) for i in text])
    
    def _alpha_count(self, text):
        return sum([1 for i in text if i.is_alpha])
    
    def _stop_word_count(self, text):
        return sum([1 for i in text if i.is_stop])
    
    def _punct_count(self, text):
        return sum([1 for i in text if i.is_punct])
    
    def _digit_count(self, text):
        return sum([1 for i in text if i.like_num])
    
    def _alpha_num_count(self, text):
        return sum([1 for i in text if i.text.isalpha()])
    
    def _first_character_count(self, text):
        return sum([1 for i in text if i.text.isalpha() and i.text[0].isupper()])
    
    def _is_non_alphabet_token(self, token):
        return not any(c.isalpha() for c in token)
    
    def _not_alphabet_count(self, text):
        return sum([1 for i in text if self._is_non_alphabet_token(i.text)])
    
    def _number_of_tokens(self, text):
        return len([1 for i in text])
    
    def _upper_case_tokens(self, text):
        return sum([1 for i in text if i.text.isupper()])
    
    def _upper_case_characters(self, text):
        match = re.findall(r'[A-Z]',text)
        return len(match)/len(text)
    
    def _more_then_avg_length(self, text):
        avg = self._avg_token_lengh(text)
        return sum([1 for i in text if len(i.text) >= avg])
    
    def _dot_count(self, text):
        return len(re.findall('\.',text))
    
    def _dolar_count(self, text):
        return len(re.findall('\$',text.text))
    
    def _hash_count(self,text):
        return len(re.findall(':',text))
    
    def _noun_count(self, text):
        return len([1 for i in text if i.pos_ in ['NOUN','PROPN']])    
    
    def  _rule_based_address(self, text):
        match = re.findall(r"(?=("+'|'.join(can_address_words)+r"))",text, re.IGNORECASE)
        return len(match)
    
    def get_feature(self, text):
        return_dict = {}
        text = nlp(text)
        tokens = len([i for i in text])
        return_dict['text'] = text.text
        return_dict['avg_token_length'] = self._avg_token_lengh(text)
        return_dict['more_then_avg'] = self._more_then_avg_length(text)/tokens
        return_dict['alpha_count'] = self._alpha_count(text)/tokens
        return_dict['stop_word_count'] = self._stop_word_count(text)/tokens
        return_dict['punct_count'] = self._punct_count(text)
        return_dict['digit_count'] = self._digit_count(text)
        return_dict['alpha_num_count'] = self._alpha_num_count(text)/tokens
        return_dict['number_of_title_tokens'] = self._first_character_count(text)/tokens
        return_dict['not_alphabet_count'] = self._not_alphabet_count(text)/tokens
        return_dict['number_of_tokens'] = self._number_of_tokens(text)
        return_dict['dolar_count'] = self._dolar_count(text)
        return_dict['dot_count'] = self._dot_count(text.text)
        return_dict['hash_count'] = self._hash_count(text.text)
        return_dict['noun_count'] = self._noun_count(text)
        return_dict['upper_case'] = self._upper_case_characters(text.text)
        return_dict['address_key_words'] = self._rule_based_address(text.text)
        return return_dict
    