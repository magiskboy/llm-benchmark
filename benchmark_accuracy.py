import os
import sys
import json
import time
import pandas as pd
import Levenshtein
import httpx
from concurrent.futures import ThreadPoolExecutor


top_p = 0.95
top_k = 40
temperature = 0.2
limit_items = 2000
openai_api_base_url = 'http://litellm-litellm-1:4000/v1/'
model = 'Qwen/Qwen2.5-Coder-1.5B'
n_cores = 8
max_tokens = 50

def main(args: list[str]):
    st_time = time.time()
    try:
        dataset_name = args[0]
        dataset_path = args[1]
    except Exception:
        print_usage()
        exit(0)

    client = httpx.Client(base_url=openai_api_base_url, headers={"Authorization": "Bearer sk-1234"})

    config_path = os.path.join(dataset_path, 'config.json')
    config = load_config_dataset(config_path)

    python_dat = os.path.join(dataset_path, 'python.jsonl')
    dataset = load_dataset(python_dat) 
    report_python = benchmark(client, dataset[:limit_items], config)
    export_report(report_python, dataset_name, 'python')
    print_summary_report(report_python, dataset_name, 'python')

    java_dat = os.path.join(dataset_path, 'java.jsonl') 
    dataset = load_dataset(java_dat) 
    report_java = benchmark(client, dataset[:limit_items], config)
    export_report(report_java, dataset_name, 'java')
    print_summary_report(report_java, dataset_name, 'java')

    f_time = time.time()
    print('Time to process', f_time - st_time, 'seconds')


def load_config_dataset(filename):
    with open(filename, 'r') as fi:
        config = json.load(fi)
        return config


def load_dataset(filename: str):
    dataset = []
    with open(filename, 'r') as fi:
        for line in fi.readlines():
            try:
                item = json.loads(line)
            except Exception:
                ...
            else:
                dataset.append(item)
    return dataset

def process_row(prompt, expected, client, config):
    payload = dict(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        stop=config.get('stop_token', ['\n']),
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        res = client.post(url='/chat/completions', json=payload, timeout=60*5)
        data = res.json()
        time_to_process = (float(res.headers['x-litellm-response-duration-ms']) - float(res.headers['x-litellm-overhead-duration-ms'])) / 1000
        prompt_tokens = data['usage']['prompt_tokens']
        completion_tokens = data['usage']['completion_tokens']
    except Exception as e:
        print(e)
        return None

    return dict(
        prompt=prompt,
        actual=data['choices'][0]['message']['content'], #type:ignore for litellm
        expected=expected,
        es_score=0,
        em_score=0,
        time_to_process=time_to_process,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

def benchmark(client: httpx.Client, dataset: list, config: dict) -> pd.DataFrame:
    report_data = []
    fs = []
    with ThreadPoolExecutor(max_workers=n_cores) as pool:
        for item in dataset:
            prompt = item.get('prompt')
            expected = item.get('groundtruth')
            if not (prompt and expected):
                continue

            future = pool.submit(process_row, prompt, expected, client, config)
            fs.append(future)
        pool.shutdown(wait=True)

    for f in fs:
        add_to_report(
            report_data,
            **f.result(),
        )

    report = pd.DataFrame(data=report_data)
    report = report.apply(calculate_score, axis=1)

    return report #type:ignore


def add_to_report(report_data: list, **kwargs):
    r = {
        **kwargs,
    }
    report_data.append(r)


def calculate_score(row):
    row['es_score'] = Levenshtein.ratio(row['expected'], row['actual'])
    row['em_score'] = int(row['expected'] == row['actual'])
    row['tps'] = (row['completion_tokens'] + row['prompt_tokens']) / row['time_to_process']
    return row


def export_report(report: pd.DataFrame, dataset_name: str, lang: str, outdir=None):
    if outdir is None:
        outdir = '.'

    filename = os.path.join(outdir, f'{dataset_name}_{lang}.xlsx')
    report.to_excel(filename, index=False)


def print_summary_report(report: pd.DataFrame, dataset_name: str, lang: str):
    title = f'Summary [{dataset_name} - {lang}]'
    print(title)
    print(f'Total items: {len(report)}')

    es_score = report['es_score'].mean()
    print(f'Mean of Edit similarity (Levenshtein): {es_score}')

    em_score = report['em_score'].mean()
    print(f'Mean of Exact math: {em_score}')

    mean_tps = report['tps'].mean()
    print(f'Tokens per Second: {mean_tps}')


def print_usage():
    print(
"""
Usage:
    python main.py <dataset_name> <dataset_path>

Arguments:
    dataset_name    Tên dataset để gắn nhãn report xuất ra (ví dụ: "qwencoder").
    dataset_path    Đường dẫn thư mục chứa file dataset, yêu cầu có:
                        - python.jsonl
                        - java.jsonl

Ví dụ:
    python main.py qwencoder ./datasets/humaneval
""")


if __name__ == '__main__':
    main(sys.argv[1:])
