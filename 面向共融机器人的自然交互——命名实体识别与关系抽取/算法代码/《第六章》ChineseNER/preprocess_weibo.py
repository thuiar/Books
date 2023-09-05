import numpy as np
from string import digits



rNUM = '(-|\+)?\d+((\.)\d+)?%?'
rENG = '[A-Za-z_.]+'
vector = []
vector_pinyin = []
vector_wubi = []
word2id = {}
pinyin2id = {}
wubi2id = {}
id2word = {}
id2pinyin = {}
id2wubi = {}
tag_id = {}
id_tag={}
word_dim=100


def load_embedding():
    print ('reading chinese word embedding.....')
    f = open('./datasets/embed.txt','r',encoding='utf-8')
    print('reading pinyin embedding.....')
    f_pinyin = open('./datasets/pinyin_vec.txt', 'r', encoding='utf-8')
    print('reading wubi embedding.....')
    f_wubi = open('./datasets/wubi_vec.txt', 'r', encoding='utf-8')
    while True:
        content=f.readline()
        if content=='':
            break
        else:
            content=content.strip().split()
            word2id[content[0]]=len(word2id)
            id2word[len(id2word)] = content[0]
            content = content[1:]
            content = [float(i) for i in content]
            vector.append(content)
    f.close()
    word2id['padding'] = len(word2id)
    word2id['unk'] = len(word2id)
    vector.append(np.zeros(shape=100, dtype=np.float32))
    vector.append(np.random.normal(loc=0.0, scale=0.1, size=100))
    id2word[len(id2word)] = 'padding'
    id2word[len(id2word)] = 'unk'
    while True:
        content_pinyin = f_pinyin.readline()
        if content_pinyin=='':
            break
        else:
            content_pinyin = content_pinyin.strip().split()
            pinyin2id[content_pinyin[0]] = len(pinyin2id)
            id2pinyin[len(id2pinyin)] = content_pinyin[0]
            content_pinyin = content_pinyin[1:]
            content_pinyin = [float(i) for i in content_pinyin]
            vector_pinyin.append(content_pinyin)
    f_pinyin.close()
    pinyin2id['padding'] = len(pinyin2id)
    pinyin2id['unk'] = len(pinyin2id)
    vector_pinyin.append(np.zeros(shape=100, dtype=np.float32))
    vector_pinyin.append(np.random.normal(loc=0.0, scale=0.1, size=100))
    id2pinyin[len(id2pinyin)] = 'padding'
    id2pinyin[len(id2pinyin)] = 'unk'
    while True:
        content_wubi = f_wubi.readline()
        if content_wubi == '':
            break
        else:
            content_wubi = content_wubi.strip().split()
            wubi2id[content_wubi[0]] = len(wubi2id)
            id2wubi[len(id2wubi)] = content_wubi[0]
            content_wubi = content_wubi[1:]
            content_wubi = [float(i) for i in content_wubi]
            vector_wubi.append(content_wubi)
    f_wubi.close()
    wubi2id['padding'] = len(wubi2id)
    wubi2id['unk'] = len(wubi2id)
    vector_wubi.append(np.zeros(shape=100, dtype=np.float32))
    vector_wubi.append(np.random.normal(loc=0.0, scale=0.1, size=100))
    id2wubi[len(id2wubi)] = 'padding'
    id2wubi[len(id2wubi)] = 'unk'


def process_train_data():
    print ('reading train data.....')
    train_word=[]
    train_label=[]
    train_length=[]
    train_wubi = []
    train_pinyin = []
    f=open('./datasets/weibo/train.txt','r',encoding='UTF-8')
    train_word.append([])
    train_label.append([])
    train_wubi.append([])
    train_pinyin.append([])
    train_max_len=0
    while True:
        content=f.readline()
        if content=='':
            break
        elif content=='\n':
            length=len(train_word[len(train_word)-1])
            train_length.append(length)
            if length>train_max_len:
                train_max_len=length
            train_word.append([])
            train_label.append([])
            train_wubi.append([])
            train_pinyin.append([])
        else:
            content=content.replace('\n','').replace('\r','').strip().split()
            remove_digits = str.maketrans('', '', digits)
            content[0] = content[0].translate(remove_digits)
            if content[0]=='':
                continue
            if content[-1]!='O':
                label1=content[-1].split('.')[0]
                label2=content[-1].split('.')[1]
                content[-1]=label1
                if label2=='NOM':
                    content[-1]='O'
            if content[0] not in word2id:
                word2id[content[0]]=len(word2id)
                vector.append(np.random.normal(loc=0.0,scale=0.1,size=100))
                id2word[len(id2word)]=content[0]
            if content[2] not in wubi2id:
                wubi2id[content[2]]=len(wubi2id)
                vector_wubi.append(np.random.normal(loc=0.0,scale=0.1,size=100))
                id2wubi[len(id2wubi)]=content[2]
            if content[1] not in pinyin2id:
                pinyin2id[content[1]]=len(pinyin2id)
                vector_pinyin.append(np.random.normal(loc=0.0,scale=0.1,size=100))
                id2pinyin[len(id2pinyin)]=content[1]
            if content[-1] not in tag_id:
                tag_id[content[-1]]=len(tag_id)
                id_tag[len(id_tag)]=content[-1]
            train_word[len(train_word)-1].append(word2id[content[0]])
            train_wubi[len(train_wubi) - 1].append(wubi2id[content[2]])
            train_pinyin[len(train_pinyin) - 1].append(pinyin2id[content[1]])
            train_label[len(train_label)-1].append(tag_id[content[-1]])

    if len(train_word[len(train_word)-1])!=0:
        train_length.append((len(train_word[len(train_word)-1])))
    if [] in train_word:
        train_word.remove([])
    if [] in train_wubi:
        train_wubi.remove([])
    if [] in train_pinyin:
        train_pinyin.remove([])
    if [] in train_label:
        train_label.remove([])
    assert len(train_word)==len(train_label)
    assert len(train_word) == len(train_wubi)
    assert len(train_word) == len(train_pinyin)
    assert len(train_word)==len(train_length)
    for i in range(len(train_word)):
        if len(train_word[i])<=train_max_len:
            for j in range(train_max_len-train_length[i]):
                train_word[i].append(word2id['padding'])
                train_wubi[i].append(wubi2id['padding'])
                train_pinyin[i].append(pinyin2id['padding'])
                train_label[i].append(tag_id['O'])
        else:
            train_word[i]=train_word[i][:train_max_len]
            train_wubi[i] = train_wubi[i][:train_max_len]
            train_pinyin[i] = train_pinyin[i][:train_max_len]
            train_label[i]=train_label[i][:train_max_len]
    train_word = np.asarray(train_word)
    train_wubi = np.asarray(train_wubi)
    train_pinyin = np.asarray(train_pinyin)
    train_label = np.asarray(train_label)
    train_length = np.asarray(train_length)
    print(train_max_len)
    np.save('./datasets/weibo/weibo_train_word.npy',train_word)
    np.save('./datasets/weibo/weibo_train_wubi.npy', train_wubi)
    np.save('./datasets/weibo/weibo_train_pinyin.npy', train_pinyin)
    np.save('./datasets/weibo/weibo_train_label.npy',train_label)
    np.save('./datasets/weibo/weibo_train_length.npy', train_length)
    return train_max_len
def process_test_data(max):
    print ('reading test data.....')
    test_word=[]
    test_label=[]
    test_pinyin = []
    test_length=[]
    test_wubi=[]
    f=open('./datasets/weibo/test.txt','r',encoding='utf-8')
    test_word.append([])
    test_label.append([])
    test_wubi.append([])
    test_pinyin.append([])
    test_max_len=0
    while True:
        content=f.readline()
        if content=='':
            break
        elif content=='\n':
            length = len(test_word[len(test_word)-1])
            test_length.append(length)
            if length>test_max_len:
                test_max_len=length

            test_word.append([])
            test_label.append([])
            test_wubi.append([])
            test_pinyin.append([])
        else:
            content = content.replace('\n', '').replace('\r', '').strip().split()
            remove_digits = str.maketrans('', '', digits)
            content[0] = content[0].translate(remove_digits)
            if content[0] == '':
                continue
            if content[-1]!='O':
                label1=content[-1].split('.')[0]
                label2=content[-1].split('.')[1]
                content[-1]=label1
                if label2=='NOM':
                    content[-1]='O'
            if content[0] not in word2id:
                word2id[content[0]]=len(word2id)
                vector.append(np.random.normal(loc=0.0,scale=0.1,size=100))
                id2word[len(id2word)]=content[0]
            if content[2] not in wubi2id:
                wubi2id[content[2]]=len(wubi2id)
                vector_wubi.append(np.random.normal(loc=0.0,scale=0.1,size=100))
                id2wubi[len(id2wubi)]=content[2]
            if content[1] not in pinyin2id:
                pinyin2id[content[1]] = len(pinyin2id)
                vector_pinyin.append(np.random.normal(loc=0.0, scale=0.1, size=100))
                id2pinyin[len(id2pinyin)] = content[1]
            if content[-1] not in tag_id:
                tag_id[content[-1]]=len(tag_id)
                id_tag[len(id_tag)]=content[-1]
            test_word[len(test_word)-1].append(word2id[content[0]])
            test_wubi[len(test_wubi)-1].append(wubi2id[content[2]])
            test_pinyin[len(test_pinyin) - 1].append(pinyin2id[content[1]])
            test_label[len(test_label)-1].append(tag_id[content[-1]])
    if len(test_word[len(test_word)-1])!=0:
        test_length.append(len(test_word[len(test_word)-1]))
    if [] in test_word:
        test_word.remove([])
    if [] in test_wubi:
        test_wubi.remove([])
    if [] in test_pinyin:
        test_pinyin.remove([])
    if [] in test_label:
        test_label.remove([])
    assert len(test_word) == len(test_label)
    assert len(test_word) == len(test_length)
    assert len(test_word) == len(test_wubi)
    assert len(test_word) == len(test_pinyin)
    for i in range(len(test_word)):
        if len(test_word[i]) <= max:
            for j in range(max - test_length[i]):
                test_word[i].append(word2id['padding'])
                test_wubi[i].append(wubi2id['padding'])
                test_pinyin[i].append(pinyin2id['padding'])
                test_label[i].append(tag_id['O'])
        else:
            test_word[i]=test_word[i][:max]
            test_wubi[i] = test_wubi[i][:max]
            test_pinyin[i] = test_pinyin[i][:max]
            test_label[i]=test_label[i][:max]
    test_word = np.asarray(test_word)
    test_wubi = np.asarray(test_wubi)
    test_pinyin = np.asarray(test_pinyin)
    test_label = np.asarray(test_label)
    test_length = np.asarray(test_length)
    print(test_max_len)
    np.save('./datasets/weibo/weibo_test_word.npy',test_word)
    np.save('./datasets/weibo/weibo_test_wubi.npy', test_wubi)
    np.save('./datasets/weibo/weibo_test_pinyin.npy', test_pinyin)
    np.save('./datasets/weibo/weibo_test_label.npy',test_label)
    np.save('./datasets/weibo/weibo_test_length.npy', test_length)

def process_dev_data(max):
    print ('reading dev data.....')
    dev_word=[]
    dev_wubi = []
    dev_pinyin = []
    dev_label=[]
    dev_length=[]
    f=open('./datasets/weibo/dev.txt','r',encoding='utf-8')
    dev_word.append([])
    dev_wubi.append([])
    dev_pinyin.append([])
    dev_label.append([])
    dev_max_len=0
    while True:
        content=f.readline()
        if content=='':
            break
        elif content=='\n':
            length = len(dev_word[len(dev_word)-1])
            dev_length.append(length)
            if length>dev_max_len:
                dev_max_len=length

            dev_word.append([])
            dev_wubi.append([])
            dev_label.append([])
            dev_pinyin.append([])
        else:
            content = content.replace('\n', '').replace('\r', '').strip().split()
            remove_digits = str.maketrans('', '', digits)
            content[0] = content[0].translate(remove_digits)
            if content[0] == '':
                continue
            if content[-1]!='O':
                label1=content[-1].split('.')[0]
                label2=content[-1].split('.')[1]
                content[-1]=label1
                if label2=='NOM':
                    content[-1]='O'
            if content[0] not in word2id:
                word2id[content[0]]=len(word2id)
                vector.append(np.random.normal(loc=0.0,scale=0.1,size=100))
                id2word[len(id2word)]=content[0]
            if content[2] not in wubi2id:
                wubi2id[content[2]]=len(wubi2id)
                vector_wubi.append(np.random.normal(loc=0.0,scale=0.1,size=100))
                id2wubi[len(id2wubi)]=content[2]
            if content[1] not in pinyin2id:
                pinyin2id[content[1]] = len(pinyin2id)
                vector_pinyin.append(np.random.normal(loc=0.0, scale=0.1, size=100))
                id2pinyin[len(id2pinyin)] = content[1]
            if content[-1] not in tag_id:
                tag_id[content[-1]]=len(tag_id)
                id_tag[len(id_tag)]=content[-1]
            dev_word[len(dev_word)-1].append(word2id[content[0]])
            dev_wubi[len(dev_wubi)-1].append(wubi2id[content[2]])
            dev_pinyin[len(dev_pinyin) - 1].append(pinyin2id[content[1]])
            dev_label[len(dev_label)-1].append(tag_id[content[-1]])
    if len(dev_word[len(dev_word)-1])!=0:
        dev_length.append(len(dev_word[len(dev_word)-1]))
    if [] in dev_word:
        dev_word.remove([])
    if [] in dev_wubi:
        dev_wubi.remove([])
    if [] in dev_pinyin:
        dev_pinyin.remove([])
    if [] in dev_label:
        dev_label.remove([])
    assert len(dev_word) == len(dev_label)
    assert len(dev_word) == len(dev_length)
    assert len(dev_word) == len(dev_pinyin)
    assert len(dev_word) == len(dev_wubi)
    for i in range(len(dev_word)):
        if len(dev_word[i]) <= max:
            for j in range(max - dev_length[i]):
                dev_word[i].append(word2id['padding'])
                dev_wubi[i].append(wubi2id['padding'])
                dev_pinyin[i].append(pinyin2id['padding'])
                dev_label[i].append(tag_id['O'])
        else:
            dev_word[i]=dev_word[i][:max]
            dev_pinyin[i] = dev_pinyin[i][:max]
            dev_wubi[i] = dev_wubi[i][:max]
            dev_label[i]=dev_label[i][:max]
    dev_word = np.asarray(dev_word)
    dev_wubi = np.asarray(dev_wubi)
    dev_pinyin = np.asarray(dev_pinyin)
    dev_label = np.asarray(dev_label)
    dev_length = np.asarray(dev_length)
    print(dev_max_len)
    np.save('./datasets/weibo/weibo_dev_word.npy',dev_word )
    np.save('./datasets/weibo/weibo_dev_wubi.npy', dev_wubi)
    np.save('./datasets/weibo/weibo_dev_pinyin.npy', dev_pinyin)
    np.save('./datasets/weibo/weibo_dev_label.npy',dev_label)
    np.save('./datasets/weibo/weibo_dev_length.npy', dev_length)


#setting=Config()
load_embedding()
max=process_train_data()
process_test_data(max)
process_dev_data(max)
vector=np.asarray(vector)
np.save('./datasets/weibo/weibo_vector.npy',vector)
vector_wubi=np.asarray(vector_wubi)
np.save('./datasets/weibo/weibo_vector_wubi.npy',vector_wubi)
vector_pinyin=np.asarray(vector_pinyin)
np.save('./datasets/weibo/weibo_vector_pinyin.npy',vector_pinyin)

f_id2word = open('./datasets/weibo/id2word.txt', 'w',encoding='UTF-8')
f_id2word.write(str(id2word))
f_id2word.close()

f_id_tag= open('./datasets/weibo/id_tag.txt', 'w',encoding='UTF-8')
f_id_tag.write(str(id_tag))
f_id_tag.close()

print ('The number of word is:')
print (len(word2id))
print ('The number of wubi is:')
print (len(wubi2id))
print ('The number of pinyin is:')
print (len(pinyin2id))
print ('The number of tag is:')
print (len(tag_id))


