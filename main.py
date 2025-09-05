import os
import sys
import json
import pandas as pd
import Levenshtein
from llama_cpp import Llama, llama_perf_context
from huggingface_hub import hf_hub_download


repo_id = "unsloth/Qwen2.5-Coder-3B-Instruct-GGUF"
model_filename = "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"
top_p = 0.95
top_k = 40
temperature = 0.2
n_ctx = 0
limit_items = 1
verbose = False


def main(args: list[str]):
    try:
        dataset_name = args[0]
        dataset_path = args[1]
    except Exception:
        print_usage()
        exit(0)

    model = load_model(
        repo_id=repo_id,
        filename=model_filename,
    )

    python_dat = os.path.join(dataset_path, 'python.jsonl')
    dataset = load_dataset(python_dat) 
    report_python = benchmark(model, dataset[:limit_items])
    export_report(report_python, dataset_name, 'python')
    print_summary_report(report_python, dataset_name, 'python')

    java_dat = os.path.join(dataset_path, 'java.jsonl') 
    dataset = load_dataset(java_dat) 
    report_java = benchmark(model, dataset[:limit_items])
    export_report(report_java, dataset_name, 'java')
    print_summary_report(report_java, dataset_name, 'java')


def load_model(repo_id: str, filename: str) -> Llama:
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        verbose=verbose,
    )
    return model


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


def benchmark(model: Llama, dataset: list) -> pd.DataFrame:
    report_data = []
    for item in dataset:
        prompt = item.get('prompt')
        expected = item.get('groundtruth')
        if not (prompt and expected):
            continue
        
        res = model.create_completion(
            prompt=prompt,
            stop=['\n'],
            echo=False,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        add_to_report(
            perf_ctx=llama_perf_context(model.ctx), 
            report_data=report_data,
            prompt=prompt,
            actual=res['choices'][0]['text'], #type:ignore
            expected=expected,
            es_score=0,
            em_score=0,
        )

    report = pd.DataFrame(data=report_data)
    report = report.apply(calculate_score, axis=1)
    return report #type:ignore


def add_to_report(perf_ctx, report_data: list, **kwargs):
    r = {
        "load_time": perf_ctx.t_load_ms,
        "time_prompt_eval": perf_ctx.t_p_eval_ms,
        "time_eval": perf_ctx.t_eval_ms,
        "prompt_tokens": perf_ctx.n_p_eval,
        "eval_tokens": perf_ctx.n_eval,
        "reused_tokens": perf_ctx.n_reused,
        **kwargs,
    }
    r['prompt_tokens_per_second'] = r['prompt_tokens'] / r['time_prompt_eval'] * 1000 
    r['eval_tokens_per_second'] = r['eval_tokens'] / r['time_eval'] * 1000
    report_data.append(r)


def calculate_score(row):
    row['es_score'] = Levenshtein.ratio(row['expected'], row['actual'])
    row['em_score'] = int(row['expected'] == row['actual'])
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

    tps = (((report["prompt_tokens"] + report["eval_tokens"]) / (report["time_eval"] + report["time_prompt_eval"])).mean()) * 1000
    print(f'Tokens per Second: {tps}')


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

