import sys
import json
import pandas as pd

name = 'steam'
if len(sys.argv) > 1:
    name = sys.argv[1]

item_cate = {}
item_map = {}
cate_map = {}
with open('./data/%s_data/%s_item_map.txt' % (name, name), 'r') as f:
    for line in f:
        conts = line.strip().split(',')
        item_map[conts[0]] = conts[1]

if name == 'taobao':
    with open('UserBehavior.csv', 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            iid = conts[1]
            if conts[3] != 'pv':
                continue
            cid = conts[2]
            if iid in item_map:
                if cid not in cate_map:
                    cate_map[cid] = len(cate_map) + 1
                item_cate[item_map[iid]] = cate_map[cid]
elif name == 'book':
    with open('./data/meta_Books.json', 'r') as f:
        for line in f:
            r = eval(line.strip())
            iid = r['asin']
            cates = r['category']
            if iid not in item_map:
                continue
            cate = cates[-1] if len(cates) > 0 else "unkown"
            if cate not in cate_map:
                cate_map[cate] = len(cate_map) + 1
            item_cate[item_map[iid]] = cate_map[cate]
elif name == 'movielens':
    with open('./data/movies.csv', 'rb') as f:
        for line in f:
            conts = str(line).strip().split(',"')
            if "movie_id" in conts[0]:
                continue
            iid = conts[0][2:]
            genres = conts[2]
            # use first genre as single category
            cid = genres[:genres.index('|')] if '|' in genres else genres[:-4]
            if iid in item_map:
                if cid not in cate_map:
                    cate_map[cid] = len(cate_map) + 1
                item_cate[item_map[iid]] = cate_map[cid]
elif name == 'steam':
    steam_df = pd.read_csv('./data/steam_raw/steam_games.csv', skiprows=1, usecols=[0,13])
    for index, row in steam_df.iterrows():
        try:
            iid = row[0].split('/')[4]
        except:
            iid = 0
        genres = str(row[1])
        cid = genres.split(',')[0]
        if iid in item_map:
            if cid not in cate_map:
                cate_map[cid] = len(cate_map) + 1
            item_cate[item_map[iid]] = cate_map[cid]

with open('./data/%s_data/%s_cate_map.txt' % (name, name), 'w') as f:
    for key, value in cate_map.items():
        f.write('%s,%s\n' % (key, value))
with open('./data/%s_data/%s_item_cate.txt' % (name, name), 'w') as f:
    for key, value in item_cate.items():
        f.write('%s,%s\n' % (key, value))
