import numpy
import json
import random
import numpy as np


class DataIterator:

    def __init__(self, source,
                 batch_size=128,
                 maxlen=100,
                 train_flag=0
                ):
        self.read(source)
        self.users = list(self.users)
        
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = maxlen
        self.index = 0

    def __iter__(self):
        return self
    
    def next(self):
        return self.__next__()

    def read(self, source):
        self.graph = {}
        self.item_graph = {}
        self.users = set()
        self.items = set()
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                time_stamp = int(conts[2])
                self.users.add(user_id)
                self.items.add(item_id)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((item_id, time_stamp))
                if item_id not in self.item_graph:
                    self.item_graph[item_id] = []
                self.item_graph[item_id].append((user_id, time_stamp))
        for user_id, value in self.graph.items():
            value.sort(key=lambda x: x[1])
            self.graph[user_id] = [x[0] for x in value]
        for item_id, value in self.item_graph.items():
            value.sort(key=lambda x: x[1]) 
            self.item_graph[item_id] = [x[0] for x in value]
        self.users = list(self.users)
        self.items = list(self.items)

    def __next__(self):
        if self.train_flag == 0:
            user_id_list = random.sample(self.users, self.batch_size)
        else:
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index+self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []
        hist_item_list = []
        hist_mask_list = []
        neighbor_hist_item_list = []
        neighbor_hist_mask_list = []
        all_neighbor_hist_item_list = []
        all_neighbor_hist_mask_list = []
        cur_neighbor_hist_item_list = []
        current_list = []
        for user_id in user_id_list:
            item_list = self.graph[user_id]
            if self.train_flag == 0:
                k = random.choice(range(4, len(item_list)))
                item_id_list.append(item_list[k])
            else:
                k = int(len(item_list) * 0.8)
                item_id_list.append(item_list[k:])
            if k >= self.maxlen:
                current_list = item_list[k-self.maxlen: k]
                hist_item_list.append(current_list)
                hist_mask_list.append([1.0] * self.maxlen)
            else:
                current_list = item_list[:k] + [0] * (self.maxlen - k)
                hist_item_list.append(current_list)
                hist_mask_list.append([1.0] * k + [0.0] * (self.maxlen - k))
            end = len(current_list) - 1
            if 0 in current_list:
                end = current_list.index(0) - 1
            start = end - 5
            if start < 0:
                start = 0
            if end < start:
                end = start
            for hist_item in current_list[start : end]:
                neighbor_list = self.item_graph[int(hist_item)]
                n_start = 0
                n_end = min(len(neighbor_list),19)
                for neighbor_user_id in neighbor_list[n_start:n_end]:
                    # TODO: avoid self-data leakage
                    if neighbor_user_id == user_id:
                        continue
                    neighbor_item_list = self.graph[neighbor_user_id]
                    neighbor_k = neighbor_item_list.index(hist_item)                    
                    if neighbor_k + self.maxlen >= int(len(neighbor_item_list) * 0.8):
                        # print(neighbor_item_list[neighbor_k : int(len(neighbor_item_list) * 0.8)])
                        neighbor_hist_item_list = neighbor_item_list[neighbor_k : int(len(neighbor_item_list) * 0.8)] + [0] * (self.maxlen - (int(len(neighbor_item_list) * 0.8 - neighbor_k)))
                    else:
                        neighbor_hist_item_list = neighbor_item_list[neighbor_k : (neighbor_k+self.maxlen)]
                    if not hist_item in neighbor_hist_item_list:
                        continue
                    end = neighbor_hist_item_list.index(hist_item) + 5
                    if end >= len(neighbor_hist_item_list):
                        end = len(neighbor_hist_item_list) - 1
                    cur_neighbor_hist_item_list += neighbor_item_list[neighbor_hist_item_list.index(hist_item):end]
            all_neighbor_k = int(len(cur_neighbor_hist_item_list) * 0.8)
            if all_neighbor_k >= self.maxlen:
                all_neighbor_hist_item_list.append(cur_neighbor_hist_item_list[all_neighbor_k-self.maxlen: all_neighbor_k])
                all_neighbor_hist_mask_list.append([1.0] * self.maxlen)
            else:
                all_neighbor_hist_item_list.append(cur_neighbor_hist_item_list[:all_neighbor_k] + [0] * (self.maxlen - all_neighbor_k))
                all_neighbor_hist_mask_list.append([1.0] * all_neighbor_k + [0.0] * (self.maxlen - all_neighbor_k))                 
        return (user_id_list, item_id_list), (hist_item_list, hist_mask_list), (all_neighbor_hist_item_list, all_neighbor_hist_mask_list)
