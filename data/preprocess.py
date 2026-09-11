import gzip
import json
import os


LANGUAGES = ['ruby', 'go', 'java', 'javascript', 'php', 'python']


for language in LANGUAGES:
    print(language)

    train = []
    valid = []
    test = []
    codebase = []

    # Find CodeSearchNet files under <language>/final.
    for root, dirs, files in os.walk(os.path.join(language, 'final')):
        for file in files:
            temp = os.path.join(root, file)

            if not (file.endswith('.jsonl') or file.endswith('.jsonl.gz')):
                continue

            if 'train' in file:
                train.append(temp)
            elif 'valid' in file:
                valid.append(temp)
                codebase.append(temp)
            elif 'test' in file:
                test.append(temp)
                codebase.append(temp)

    train_data = {}
    valid_data = {}
    test_data = {}
    codebase_data = {}

    # Load JSONL files and index records by URL.
    for files, data in [
        (train, train_data),
        (valid, valid_data),
        (test, test_data),
        (codebase, codebase_data),
    ]:
        for file in files:
            if file.endswith('.jsonl.gz'):
                opener = gzip.open
                mode = 'rt'
            else:
                opener = open
                mode = 'r'

            with opener(file, mode, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    js = json.loads(line)

                    if 'url' not in js:
                        continue

                    data[js['url']] = js

    # Create output directory if it does not exist.
    os.makedirs(language, exist_ok=True)

    # Generate the final files according to the URLs listed in
    # <language>/{train,valid,test,codebase}.txt.
    for tag, data in [
        ('train', train_data),
        ('valid', valid_data),
        ('test', test_data),
        ('codebase', codebase_data),
    ]:
        txt_file = os.path.join(language, f'{tag}.txt')
        jsonl_file = os.path.join(language, f'{tag}.jsonl')

        with open(txt_file, encoding='utf-8') as f2, \
             open(jsonl_file, 'w', encoding='utf-8') as f1:

            for line in f2:
                url = line.strip()

                if not url or url not in data:
                    continue

                # Copy the record so that we don't modify the
                # original object stored in the dictionary.
                js = dict(data[url])

                if tag in ['valid', 'test']:
                    js['original_string'] = ''
                    js['code'] = ''
                    js['code_tokens'] = []

                if tag == 'codebase':
                    js['docstring'] = ''
                    js['docstring_tokens'] = []

                f1.write(json.dumps(js, ensure_ascii=False) + '\n')
