import re
import attr
import random
import pandas as pd
from faker import Faker
from .utility import formating
from .CONSTATNTS import *
import nlpaug.augmenter.char as nac

fake = Faker()
aug = nac.OcrAug()


@attr.s
class DataGeneration:
    template_list = attr.ib()
    number = attr.ib()
    
    @property
    def data(self):
        return_data = []
        const_data = self.constant_generation()
        for i in range(0,self.number):
            temp, class_3_label = random.choice(self.template_list)
            class_2_label = class_3_label
            if class_3_label == 'partial-addressline':
                class_2_label = 'addressline'
            matcher = re.findall('\{.*?\}',temp)
            for i in matcher:
                key = i.replace('{','').replace('}','')
                tp = random.choice(const_data[key])
                temp = temp.replace(i, tp)
            if random.randint(0,100)%3 == 0:
                temp = aug.augment(temp, n=1)
            return_data.append((temp, class_2_label, class_3_label))
        dataframe = pd.DataFrame(data=return_data, columns=['text','class_2_target','class_3_target' ])
        return formating(dataframe)
            
    def constant_generation(self):
        n = int(self.number*(2))
        DATE = self.date_generator(n)
        NUMBER = self.number_generator(n)
        CITY_NAME = self.city_name_generator(n)
        ADDRESS = self.full_address_generator(n)
        STREET_NAME = self.street_name_generator(n)
        RANDOM_WORDS = self.random_word_generator(n)
        COMPANY_NAME = self.company_name_generator(n)
        POSTAL_NUMBER = self.postal_number_generator(n)
        STREET_ADDRESS = self.street_address_generator(n)
        BUILDING_NUMBER = self.building_number_generator(n)
        
        data = dict(COMPANY_NAME =  COMPANY_NAME,BUILDING_NUMBER = BUILDING_NUMBER,
                    STREET_NAME = STREET_NAME, CITY_NAME = CITY_NAME, NUMBER = NUMBER,
                    STREET_ADDRESS = STREET_ADDRESS, CAPITAL_WORDS = JUNK, DATE = DATE,
                    POSTAL_NUMBER = POSTAL_NUMBER, RANDOM_WORDS = RANDOM_WORDS, ADDRESS = ADDRESS)
        return data
        
    
    def company_name_generator(self, n=50):
        company_name = []
        for i in range(n):
            cmp_name = fake.company()
            if random.randint(1,10) % 5 == 0:
                cmp_name += ' ' + fake.company_suffix()
            company_name.append(cmp_name)
        return company_name

    def building_number_generator(self, n=50):
        building_name= [fake.building_number() for i in range(n)]
        return building_name

    def street_name_generator(self, n=50):
        street_name = []
        for i in range(n):
            str_name = fake.street_name()
            if random.randint(1,15)%6 == 0:
                str_name += ' '+fake.street_suffix()
            street_name.append(str_name)
        return street_name

    def street_address_generator(self, n=50):
        street_name = [fake.street_address() for i in range(n)]
        return street_name

    def city_name_generator(self, n=50):
        city_names = []
        for i in range(n):
            city_name = fake.city()
            if random.randint(1,10)%3 == 0:
                city_name = fake.city_prefix()+' '+city_name
            if random.randint(1,10)%5 == 0:
                city_name += ' '+fake.city_suffix()
            city_names.append(city_name)
        return city_names

    def postal_number_generator(self, n =50):
        return [fake.postalcode() for i in range(0,n)]

    def date_generator(self, n=50):
        prefix = ['Date:', 'date ', 'Date-', 'DATE:', 'Date: ', 'Date']
        date = [fake.date(random.choice(date_fmt)) for i in range(0,n)]
        data = [i if random.randint(0,10)%3 == 0 else random.choice(prefix)+' '+i for i in date]
        return data

    def number_generator(self,n=50):
        fmt = ['##', '####','###']
        num_list = []
        for i in range(n):
            if random.randint(0,20)%3 == 0:
                num = fake.numerify(random.choice(fmt))+random.choice(puct_list)
            elif random.randint(0,20)%7 == 0:
                _ = ['##', '#','###']
                num = '#'+fake.numerify(random.choice(_))+'#'+fake.numerify(random.choice(_))
            else:
                num = fake.numerify(random.choice(fmt))
            num_list.append(num)
        return num_list

    def word_processing(self, lst):
        _ = []
        for i in lst:
            if random.randint(1,10)%3 == 0:
                _.append(i.title())
            elif random.randint(1,10)%7 == 0:
                _.append(i.upper())
            else:
                _.append(i)
        return ' '.join(i)


    def random_word_generator(self, n=50):
        text = [self.word_processing(fake.words(random.randint(2, 7))) for i in range(n)]
        return text

    def full_address_generator(self, n=50):
        return [fake.address() for i in range(n)]
