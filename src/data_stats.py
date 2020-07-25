import attr
from fuzzywuzzy import fuzz
from .utility import full_text_matcher
from .utility import partial_text_matcher

@attr.s
class DataStats:
    """
    Utility to check a bunch of differnent statistics on the dataset.
    """
    dataset = attr.ib()
    matching = attr.ib(default='full-match')
    not_found = attr.ib(init=True,default=[])
    
    @property
    def stats(self):
        self.not_found = []
        stats = self.stats_count()
        for i,j in stats.items():
            print(f"Number of the documents where {i} are present: {j}/{len(self.dataset)}")
        
        
    def stats_count(self):
        buyer_count = 0
        vendor_count = 0
        at_least_one = 0
        both_available = 0
        multi_line_buyer = 0
        multi_line_vendor = 0
        for idx,_doc in enumerate(self.dataset):
            document = _doc['document']
            ground_truth = _doc['ground_truth']
            vendor_present, buyer_present = self._is_ground_truth_present(ground_truth, document)
            if self.matching == 'partial-match':
                if len(vendor_present) > 1:
                    multi_line_vendor +=1
                
                if len(buyer_present) > 1:
                    multi_line_buyer += 1
                vendor_present = len(vendor_present) > 0
                buyer_present = len(buyer_present) > 0
                
                
            if vendor_present and buyer_present:
                at_least_one += 1
                both_available += 1
                vendor_count += 1
                buyer_count += 1
                
            elif vendor_present:
                at_least_one += 1
                vendor_count += 1
                self.not_found.append(_doc)
                
            elif buyer_present:
                at_least_one +=1
                buyer_count += 1
                self.not_found.append(_doc)
            else:
                self.not_found.append(_doc)
                
        return_data = {'both entities':both_available,
                       'at least one': at_least_one,
                       'vendor address':vendor_count, 
                       'buyer address':buyer_count}
        if self.matching == 'partial-match':
            return_data['multi line vendor addresses'] = multi_line_vendor
            return_data['multi line buyer addresses'] = multi_line_buyer
        return return_data

        
    def _is_ground_truth_present(self, ground_truth, document):
        return_list = []
        text_data = [i['text'] for i in document]
        for i,j in ground_truth.items():
            if self.matching == 'full-match':
                return_list.append(full_text_matcher(j,text_data))
            else:
                return_list.append(partial_text_matcher(j, text_data))
        return tuple(return_list)
    

    
def is_same(actual, ref, check_for):
    same = True
    for i in check_for:
        if fuzz.ratio(actual[i]['text'], ref[i]['text']) < 70:
            same = False
            break
    return same

def check_repeated_data(dataset, check_for = ['vendor_address', 'buyer_address']):
    repeated_data = {}
    for index_1, doc_1 in enumerate(dataset):
        repeated_doc_id = []
        ground_truth_1 = doc_1['ground_truth']
        for index_2, doc_2 in enumerate(dataset):
            if index_2 == index_1:
                continue
            ground_truth_2 = doc_2['ground_truth']
            if is_same(ground_truth_1, ground_truth_2, check_for):
                repeated_doc_id.append(index_2)
        if len(repeated_doc_id):
            repeated_data[index_1] = repeated_doc_id
    return repeated_data