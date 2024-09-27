# 코드 참고 : https://github.com/alexa/alexa-with-dstc9-track1-dataset/blob/master/baseline/dataset.py
# Dataset 클래스에서 도메인과 개체명을 추출하는 전처리(Preprocessor) 모듈 역할을 수행
# 기존 Dataset에서 multiwoz에 있는 hotel.db, restaurant.db 등의 지역 정보(area)를 활용하여 각 외부 지식 엔티티에 area 지역 정보 추가
# 성능을 올리기 위해 전처리 모듈의 개체명 추출 성능을 올린 코드
import os
import re
import json
import logging
import itertools

import torch
import numpy as np

from tqdm import tqdm
from collections import Counter

logger = logging.getLogger(__name__)
SPECIAL_TOKENS = {
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "additional_special_tokens": [
        "[USER]",
        "[SYSTEM]",
        "[DOMAIN]",
        "[ENTITY]",
        "[TITLE]",
        "[BODY]",
        "[CITY]",
        "[NONE]"
    ]
}
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_]')

def normalize_text(text):
    normalized_text = RE_PUNC.sub(' ', text).replace('\'', "")
    normalized_text = ' '.join(normalized_text.split())

    return normalized_text.lower().strip()


class DatasetWalker():
    def __init__(self, data_root, data_type):
        if data_type not in ['train', 'valid', 'test']:
            raise ValueError('Wrong data type : %s' % (data_type))

        logs_file = os.path.join(data_root, data_type, 'logs.json')
        with open(logs_file, 'r', encoding='utf-8') as fp:
            self.logs = json.load(fp)

        labels_file = os.path.join(data_root, data_type, 'labels.json')
        with open(labels_file, 'r', encoding='utf-8') as fp:
            self.labels = json.load(fp)

    def __iter__(self):
        for log, label in zip(self.logs, self.labels):
            yield(log, label)

    def __len__(self):
        return len(self.logs)


class KnowledgeReader():
    def __init__(self, data_root):
        self.data_root = data_root
        knowledge_file = os.path.join(data_root, 'knowledge.json')
        with open(knowledge_file, 'r', encoding='utf-8') as fp:
            self.knowledge = json.load(fp)

        self.domain_token = SPECIAL_TOKENS["additional_special_tokens"][2]
        self.entity_token = SPECIAL_TOKENS["additional_special_tokens"][3]
        self.title_token = SPECIAL_TOKENS["additional_special_tokens"][4]
        self.body_token = SPECIAL_TOKENS["additional_special_tokens"][5]
        self.city_token = SPECIAL_TOKENS["additional_special_tokens"][6]
        self.none_token = SPECIAL_TOKENS["additional_special_tokens"][-1]

        self.passages = self._create_passages()
        # areas : 도시 별 지역을 저장, area_by_entity : 엔티티 별 지역을 저장
        self.areas, self.area_by_entity = self._create_areas()
        # 개체명의 sub string match를 위해 각 개체명의 서브 스트링을 저장
        # ex) 개체명 : EXPRESS BY HOLIDAY INN CAMBRIDGE, 서브 스트링 : HOLIDAY INN CAMBRIDGE
        #     서브 스트링만 나타나더라도 개체명 매칭이 되도록 하는 것이 목표
        self.entity_sub_strs = self._create_entity_sub_strs()

    def _create_areas(self):
        area_by_entity = dict()
        areas = {"cambridge": list(), "san francisco": list()}
        for domain, domain_dict in self.knowledge.items():
            if domain in ['taxi', 'train']:
                continue
            for entity_id, entity_dict in domain_dict.items():
                name = entity_dict['name']
                city = normalize_text(entity_dict['city'])
                area = normalize_text(entity_dict['area'])
                if bool(area):
                    areas[city].append(area)
                area_by_entity[name] = area

        for city in areas:
            areas[city] = list(set(areas[city]))

        return areas, area_by_entity

    def _create_entity_sub_strs(self):
        unique_sub_strs_file = os.path.join(self.data_root, 'entity_sub_strs.json')

        entity_names = list()
        all_sub_str = list()
        for area in list(itertools.chain(*list(self.areas.values()))) + \
                ["cambridge", "san francisco"]:
            all_sub_str.extend(area.split(" "))

        for domain, domain_dict in self.knowledge.items():
            if domain in ['taxi', 'train']:
                continue
            for entity_id, entity_dict in domain_dict.items():
                entity_name = entity_dict['name']
                entity_names.append(entity_name)
                entity_name = normalize_text(entity_name)
                all_sub_str.extend(entity_name.split(" "))

        counter = Counter(all_sub_str)
        all_unique_sub_str = [sub_str for sub_str in all_sub_str if counter[sub_str] == 1]
        unique_sub_strs = dict()
        for entity_name in entity_names:
            unique_sub_strs[entity_name] = list()
            splited_strs = normalize_text(entity_name).split(" ")
            for unique_sub_str in all_unique_sub_str:
                if unique_sub_str in splited_strs:
                    unique_sub_strs[entity_name].append(unique_sub_str)

        for entity_name, sub_strs in unique_sub_strs.items():
            normalized_entity_name = entity_name.lower()
            splited_entity_names = normalized_entity_name.split('/')[0].split(" - ")[0].split(', ')
            splited_entity_names = [splited_entity_name.strip() for splited_entity_name in splited_entity_names]
            three_gram_sub_strs = list()
            for splited_entity_name in splited_entity_names:
                splited_entity_name = normalize_text(splited_entity_name)
                splited_strs = splited_entity_name.split(" ")
                for sub_str in sub_strs:
                    if sub_str not in splited_strs:
                        continue
                    idx = splited_strs.index(sub_str)
                    if len(splited_strs) > 3:
                        if idx == 0:
                            three_gram_sub_strs.append(" ".join([splited_strs[0], splited_strs[1], splited_strs[2]]))
                        elif idx == len(splited_strs) - 1:
                            three_gram_sub_strs.append(" ".join([splited_strs[-3], splited_strs[-2], splited_strs[-1]]))
                        else:
                            three_gram_sub_strs.append(" ".join([splited_strs[idx-1], splited_strs[idx], splited_strs[idx+1]]))
                    else:
                        three_gram_sub_strs.append(" ".join(splited_strs))

            unique_sub_strs[entity_name] = list(set(three_gram_sub_strs))
            if not unique_sub_strs[entity_name]:
                for splited_entity_name in splited_entity_names:
                    splited_entity_name = normalize_text(splited_entity_name)
                    if splited_entity_name:
                        unique_sub_strs[entity_name].append(
                            splited_entity_name
                        )

        self.duplicate_entities = list()
        total_sub_str = [sub_str for sub_strs in unique_sub_strs.values() for sub_str in sub_strs]
        counter = Counter(total_sub_str)
        for entity_name, sub_strs in unique_sub_strs.items():
            if len(sub_strs) > 1:
                continue
            sub_str = sub_strs[0]
            if counter[sub_str] > 1:
                self.duplicate_entities.append(entity_name)

        with open(unique_sub_strs_file, 'w', encoding='utf-8') as fp:
            json.dump(unique_sub_strs, fp, indent=4, ensure_ascii=False)

        return unique_sub_strs

    def _create_passages(self):
        passages = dict()
        knowledge_docs = self.get_doc_list()
        knowledge_docs.append({
            'domain': self.none_token,
            'entity_id': '*',
            'entity_name': self.none_token,
            'doc_id': '*',
            'doc': {
                'title': self.none_token,
                'body': self.none_token
            },
            "city": self.none_token,
        })
        for doc_obj in knowledge_docs:
            key = "{}__{}__{}".format(doc_obj["domain"], doc_obj["entity_id"], doc_obj["doc_id"])

            passage = self._doc_to_passage(doc_obj)
            passages[key] = passage

        return passages

    def _doc_to_passage(self, doc_obj):
        snippets = [
            self.domain_token + doc_obj["domain"],
            self.entity_token + doc_obj["entity_name"],
            self.title_token + doc_obj["doc"]["title"],
            self.body_token + doc_obj["doc"]["body"],
            self.city_token + doc_obj["city"],
        ]

        return "".join(snippets)

    def get_domain_list(self):
        return list(self.knowledge.keys())

    def get_entity_name_list(self, domain):
        if domain not in self.get_domain_list():
            raise ValueError(f"invalid domain name : {domain}")

        entity_ids = []
        for entity_id in self.knowledge[domain].keys():
            try:
                entity_id = int(entity_id)
                entity_ids.append(entity_id)
            except:
                pass

        entity_names = []
        for entity_id in sorted(entity_ids):
            entity_name = self.knowledge[domain][str(entity_id)]['name']
            entity_names.append(entity_name)

        return entity_names

    def get_doc_list(self, domain=None, entity_id=None):
        if domain is None:
            domain_list = self.get_domain_list()
        else:
            if domain not in self.get_domain_list():
                raise ValueError("invalid domain name : %s" % domain)
            domain_list = [domain]

        doc_list = []
        for domain in domain_list:
            if entity_id is None:
                for item_id, item_obj in self.knowledge[domain].items():
                    for doc_id, doc_obj in item_obj['docs'].items():
                        document = self.get_doc(domain, item_id, doc_id)
                        doc_list.append(document)
            else:
                if str(entity_id) not in self.knowledge[domain]:
                    raise ValueError("invalid entity id : %s" % str(entity_id))

                entity_obj = self.knowledge[domain][str(entity_id)]
                for doc_id, doc_obj in entity_obj['docs'].items():
                    document = self.get_doc(domain, entity_id, doc_id)
                    doc_list.append(document)

        return doc_list

    def get_doc(self, domain, entity_id, doc_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name : %s" % domain)
        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id : %s" % str(entity_id))
        if str(doc_id) not in self.knowledge[domain][str(entity_id)]['docs']:
            raise ValueError("invalid doc id : %s" % str(doc_id))

        doc_obj = self.knowledge[domain][str(entity_id)]['docs'][str(doc_id)]
        entity_name = self.knowledge[domain][str(entity_id)]['name'] or self.none_token
        city = self.knowledge[domain][str(entity_id)]['city']
        document = {
            'domain': domain,
            'entity_id': int(entity_id) if entity_id != "*" else "*",
            'entity_name': entity_name,
            'doc_id': doc_id,
            'doc': {
                'title': doc_obj['title'],
                'body': doc_obj['body']
            },
            'city': city,
        }

        return document


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            args,
            data_root,
            data_type,
            tokenizer,
            knowledge_reader
        ):
        self.args = args
        self.data_root = data_root
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.knowledge_reader = knowledge_reader

        self.user_token = SPECIAL_TOKENS["additional_special_tokens"][0]
        self.system_token = SPECIAL_TOKENS["additional_special_tokens"][1]
        self.domain_token = SPECIAL_TOKENS["additional_special_tokens"][2]
        self.entity_token = SPECIAL_TOKENS["additional_special_tokens"][3]
        self.none_token = SPECIAL_TOKENS["additional_special_tokens"][7]

        self.dataset_walker = DatasetWalker(self.data_root, data_type)
        self.examples = self._create_examples()
        none_idx = len(self.knowledge_reader.passages) - 1
        #self.examples = [example for example in self.examples if example['label_idx'] != none_idx]

    def _create_examples(self):
        examples_file = os.path.join(
            self.data_root,
            self.data_type,
            "example_" + \
            self.args.data_language + \
            ".json"
        )
        if not(self.args.model_type == "generator" and self.data_type == "test") and os.path.isfile(examples_file):
            logger.info(f"Load {self.data_type} examples.")
            with open(examples_file, 'r', encoding='utf-8') as fp:
                return json.load(fp)

        prediction_results = [None for _ in range(len(self.dataset_walker))]
        if self.args.model_type == "generator" and self.data_type == "test":
            pred_result_file_path = self.args.retriever_prediction_result_path
            with open(pred_result_file_path, 'r', encoding='utf-8') as fp:
                 prediction_results = [json.loads(result) for result in fp]

        domains = self.knowledge_reader.get_domain_list()
        domain_entities = dict()
        for domain in domains:
            domain_entities[domain] = self.knowledge_reader.get_entity_name_list(domain)

        areas = self.knowledge_reader.areas
        areas = [area for area in areas['san francisco']]
        def extract_area(text):
            extracted_area = list()
            for area in areas:
                pattern = r"\b" + area + r"\b"
                match = re.search(pattern, normalize_text(text))
                if match is not None:
                    extracted_area.append(area)

            return extracted_area

        def has_sub_str(normalized_text, sub_strs):
            for sub_str in sub_strs:
                if sub_str in normalized_text:
                    return True

            return False

        entity_sub_strs = self.knowledge_reader.entity_sub_strs
        def extract_entity(text):
            normalized_text = normalize_text(text)
            extracted_entity = dict()
            for domain in domain_entities.keys():
                extracted_entity[domain] = [
                    entity for entity in domain_entities[domain] \
                    if has_sub_str(normalized_text, entity_sub_strs[entity])
                ]

            return extracted_entity

        duplicate_entities = self.knowledge_reader.duplicate_entities
        area_by_entity = self.knowledge_reader.area_by_entity
        def verify_extracted_entity(extracted_entity, entire_area):
            if not entire_area:
                return extracted_entity

            verified_extracted_entity = dict()
            for domain, entity_names in extracted_entity.items():
                verified_extracted_entity[domain] = list()
                for entity_name in entity_names:
                    if entity_name not in duplicate_entities:
                        verified_extracted_entity[domain].append(entity_name)
                    elif area_by_entity[entity_name] in entire_area:
                        verified_extracted_entity[domain].append(entity_name)

            return verified_extracted_entity

        with open(os.path.join("data/", "taxi_db.json"), 'r') as fp:
            taxi_types = json.load(fp)['taxi_types']
        def extract_domain(text, extracted_entity):
            extracted_domain = dict()
            for domain in domain_entities.keys():
                extracted_domain[domain] = True \
                    if domain.lower() in text.lower() or extracted_entity[domain] else False

            for taxi_type in taxi_types:
                if taxi_type.lower() in text.lower():
                    extracted_domain['taxi'] = True

            return extracted_domain

        def extract_recent_domain_entity(extracted_domains, extracted_entities):
            recent_domain = dict()
            recent_entity = dict()
            for domain in domain_entities.keys():
                recent_domain[domain] = False
                recent_entity[domain] = list()
            recent_idx = -1
            for i, (extracted_domain, extracted_entity) in enumerate(zip(reversed(extracted_domains), reversed(extracted_entities))):
                if any([value for value in extracted_domain.values()]):
                    recent_domain = extracted_domain
                    recent_entity = extracted_entity
                    recent_idx = i
                    break

            for domain in recent_domain.keys():
                if recent_domain[domain] and recent_entity[domain]:
                    return (recent_domain, recent_entity)

            recent_entity = {domain: list() for domain in recent_domain.keys()}
            for extracted_entity in list(reversed(extracted_entities))[recent_idx:]:
                for domain, entity in extracted_entity.items():
                    if not recent_entity[domain] and entity:
                        recent_entity[domain].extend(entity)

            for domain, value in recent_domain.items():
                if not value:
                    recent_entity[domain] = list()

            return (recent_domain, recent_entity)

        passages = self.knowledge_reader.passages
        examples = []
        for (log, label), result in tqdm(zip(self.dataset_walker, prediction_results), \
                desc=f"Create {self.data_type} dataset examples", \
                disable=self.args.local_rank not in [-1, 0]): # only show progress bar in one process
            key = "__".join([
                    f"{label['knowledge'][0]['domain']}",
                    f"{label['knowledge'][0]['entity_id']}",
                    f"{label['knowledge'][0]['doc_id']}"
                ]) if label["target"] else f"{self.none_token}__*__*"

            if key not in list(passages.keys()):
                raise ValueError(f"Invalid key : {key}")

            passage = passages[key]
            label_idx = list(passages.keys()).index(key)

            history = [
                (self.user_token if (len(log) - i) % 2 == 1 else self.system_token) + turn["text"]
                for i, turn in enumerate(log)
            ]

            entire_area = list()
            extracted_areas = list()
            extracted_domains = list()
            extracted_entities = list()
            for text in history:
                extracted_area = extract_area(text)
                extracted_entity = extract_entity(text)
                if extracted_area:
                    entire_area = list(set(entire_area + extracted_area))
                if self.data_type == "test":
                    extracted_entity = verify_extracted_entity(extracted_entity, entire_area)
                extracted_domain = extract_domain(text, extracted_entity)

                extracted_areas.append(extracted_area)
                extracted_entities.append(extracted_entity)
                extracted_domains.append(extracted_domain)

            pair_domains = [extracted_domains[0]]
            pair_entities = [extracted_entities[0]]
            for i in range(1, len(extracted_domains), 2):
                pair_domain = dict()
                pair_entity = dict()
                for domain in domains:
                    pair_domain[domain] = extracted_domains[i][domain] or extracted_domains[i+1][domain]
                    pair_entity[domain] = list(set(extracted_entities[i][domain] + extracted_entities[i+1][domain]))
                pair_domains.append(pair_domain)
                pair_entities.append(pair_entity)

            recent_domain, recent_entity = extract_recent_domain_entity(
                pair_domains, pair_entities
            )

            entire_domain = dict()
            for domain in domains:
                entire_domain[domain] = any(extracted_domain[domain] for extracted_domain in extracted_domains)
            entire_entity = {domain: list() for domain in entire_domain.keys()}
            for extracted_entity in extracted_entities:
                for domain, entities in extracted_entity.items():
                    entire_entity[domain].extend(entities)
            for domain, entities in entire_entity.items():
                entire_entity[domain] = list(set(entities))

            prev_example = examples[-1] if len(examples) > 0 else None
            if self.data_type != 'test' and prev_example is not None and (len(history) - len(prev_example['history'])) > 0:
                if (len(history) - len(prev_example['history']))%2 == 0 and prev_example['history'][0] == history[0]:
                    prev_example['response'] = history[len(prev_example['history'])]

            examples.append({
                "history": history,
                "response": (self.system_token + label['response']) if label['target'] else "",
                "passage": passage,
                "label_idx": label_idx,
                "pred_idx": result['pred_idx'] if result is not None else None,
                "pred_passage": list(passages.values())[result['pred_idx'][0]] if result is not None else None,
                "entire_domain": entire_domain,
                "entire_entity": entire_entity,
                "recent_domain": recent_domain,
                "recent_entity": recent_entity,
                "entire_area": entire_area,
            })

        with open(examples_file, 'w', encoding='utf-8') as fp:
            json.dump(examples, fp, indent=4, ensure_ascii=False)

        return examples

    def build_input_from_segments(self, example):
        raise NotImplementedError

    def __getitem__(self, index):
        example = self.examples[index]
        this_inst = self.build_input_from_segments(
            example
        )

        return this_inst

    def collate_fn(self, batch):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class ICTDataset(BaseDataset):
    def __init__(
            self,
            args,
            data_root,
            data_type,
            tokenizer,
            knowledge_reader,
        ):
        super().__init__(args, data_root, data_type, tokenizer, knowledge_reader)

    def build_input_from_segments(self, example):
        this_inst = dict()
        history = "".join(example['history'][:-1])
        history = self.tokenizer.decode(
            self.tokenizer.encode(history, add_special_tokens=False, truncation=True, max_length=self.args.history_max_tokens)
        )
        this_inst['query'] = example['history'][-1]
        this_inst['document'] = history + example['passage'].split("[CITY]")[0] + example['response']

        return this_inst

    def collate_fn(self, batch):
        encoded_batch = dict()
        querys = [data['query'] for data in batch]
        documents = [data['document'] for data in batch]

        encoded_querys = self.tokenizer(querys, padding=True, return_tensors='pt')
        encoded_batch['query_input_ids'] = encoded_querys['input_ids']
        encoded_batch['query_token_type_ids'] = encoded_querys['token_type_ids']
        encoded_batch['query_attention_mask'] = encoded_querys['attention_mask']
        encoded_batch['querys'] = self.tokenizer.batch_decode(encoded_batch['query_input_ids'])

        encoded_documents = self.tokenizer(documents, padding=True, return_tensors='pt')
        encoded_batch['document_input_ids'] = encoded_documents['input_ids']
        encoded_batch['document_token_type_ids'] = encoded_documents['token_type_ids']
        encoded_batch['document_attention_mask'] = encoded_documents['attention_mask']
        encoded_batch['documents'] = self.tokenizer.batch_decode(encoded_batch['document_input_ids'])

        encoded_batch['label_idxs'] = torch.tensor([i for i in range(len(batch))])

        return encoded_batch


class SelectionDataset(BaseDataset):
    def __init__(
            self,
            args,
            data_root,
            data_type,
            tokenizer,
            knowledge_reader,
        ):
        super().__init__(args, data_root, data_type, tokenizer, knowledge_reader)


    def build_input_from_segments(self, example):
        this_inst = {
            "query": None,
            "label_idx": example['label_idx'],
        }
        if self.args.input_type in ['entire', 'recent']:
            if self.args.input_type == "entire":
                history_domain = [domain for domain, value in example['entire_domain'].items() if value]
                history_entity = \
                    example['entire_entity'].get("hotel", []) + \
                    example['entire_entity'].get("restaurant", []) + \
                    example['entire_entity'].get("attraction", [])
            else:
                history_domain = [domain for domain, value in example['recent_domain'].items() if value]
                history_entity = \
                    example['recent_entity'].get("hotel", []) + \
                    example['recent_entity'].get("restaurant", []) + \
                    example['recent_entity'].get("attraction", [])

            domain = ", ".join(history_domain)
            entity = ", ".join(history_entity)

            this_inst['query'] = \
                self.domain_token + domain + \
                self.entity_token + entity + \
                "".join(example['history'][self.args.history_max_utterances*(-1):])
        elif self.args.input_type == 'history':
            history = "".join(example['history'][:-1])
            history = self.tokenizer.decode(
                self.tokenizer.encode(history, add_special_tokens=False, truncation=True, max_length=self.args.history_max_tokens)
            )
            this_inst['query'] = history + example['history'][-1]
        else:
            raise ValueError(f"Wrong input type : {self.args.input_type}")

        return this_inst

    def collate_fn(self, batch):
        encoded_batch = dict()
        querys = [data['query'] for data in batch]

        encoded_querys = self.tokenizer(querys, padding=True, return_tensors='pt')
        encoded_batch['input_ids'] = encoded_querys['input_ids']
        encoded_batch['token_type_ids'] = encoded_querys['token_type_ids']
        encoded_batch['attention_mask'] = encoded_querys['attention_mask']
        encoded_batch['querys'] = self.tokenizer.batch_decode(encoded_batch['input_ids'])

        encoded_batch['label_idxs'] = torch.tensor([data['label_idx'] for data in batch])

        return encoded_batch


class GenerationDataset(BaseDataset):
    def __init__(
            self,
            args,
            data_root,
            data_type,
            tokenizer,
            knowledge_reader,
        ):
        super().__init__(args, data_root, data_type, tokenizer, knowledge_reader)

    def build_input_from_segments(self, example):
        this_inst = {
            "query": None,
            "response": example['response'],
            "label_idx": example['label_idx'],
            "pred_idx": example['pred_idx'],
        }
        if self.data_type == "test":
            history = "".join(example['history'][:-1])
            history = self.tokenizer.decode(
                self.tokenizer.encode(history, add_special_tokens=False, truncation=True, max_length=self.args.history_max_tokens)
            )
            this_inst['query'] = history + example['history'][-1] + \
                re.sub(r".+(\[BODY\].+)\[CITY\].+", r"\1", example['pred_passage'])
        else:
            history = "".join(example['history'][:-1])
            history = self.tokenizer.decode(
                self.tokenizer.encode(history, add_special_tokens=False, truncation=True, max_length=self.args.history_max_tokens)
            )
            this_inst['query'] = history + example['history'][-1] + \
                re.sub(r".+(\[BODY\].+)\[CITY\].+", r"\1", example['passage'])

        return this_inst

    def collate_fn(self, batch):
        querys = [self.tokenizer.bos_token + data['query'] + self.tokenizer.sep_token for data in batch]
        responses = [data['response'] + self.tokenizer.eos_token for data in batch]
        label_idxs = [data['label_idx'] for data in batch]
        pred_idxs = [data['pred_idx'] for data in batch]

        if self.data_type != "test":
            encoded_batch = self.tokenizer(querys, responses, padding=True, return_tensors='pt')

            results = dict()
            results['input_ids'] = encoded_batch['input_ids']
            results['attention_mask'] = encoded_batch['attention_mask']
            results['labels'] = torch.where(
                encoded_batch['input_ids'] == self.tokenizer.pad_token_id,
                -100 * torch.ones_like(encoded_batch['input_ids']),
                encoded_batch['input_ids']
            )
            sep_idx = (results['labels'] == self.tokenizer.sep_token_id).nonzero()
            for batch_idx, index in sep_idx:
                results['labels'][batch_idx, :index.item() + 1] = -100
        else:
            encoded_batch = self.tokenizer(querys, padding=True, return_tensors='pt')

            results = dict()
            results['input_ids'] = encoded_batch['input_ids']
            results['attention_mask'] = encoded_batch['attention_mask']

        results['gener_inputs'] = querys
        results['pred_idxs'] = pred_idxs
        results['label_idxs'] = label_idxs
        results['responses'] = [data['response'] for data in batch]

        return results