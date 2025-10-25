import os
from typing import List
from collections import deque
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import torch
import pyarrow.parquet as pq
import tiktoken

##################################
# README:
# This dataloader assumes the data is already downloaded and stored in parquet files.
# and set os.environ['NANOCHAT_BASEDIR'] = '~/.cache/nanochat'
# download the fineweb data from: nanochat repo approach
# Currently not use a trained tokenizer, just use tiktoken's o200k_base

# To test the speed of loading token:
# data_loader = tokenizing_distributed_data_loader(32, 1024, 'train', tokenizer_threads=8)
# from tqdm import tqdm
#
# counter = 0
# with tqdm(total=10000) as pbar:
#     for inputs, targets in data_loader:
#         pbar.update(1)
#         if pbar.n >= 10000:
#             break
# about 80 * 32 * 1024 tokens per second on 8 threads (2.6M tokens per seconds)
##################################


def list_parquet_files(data_dir=None, num_partitions=1):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_paths = [os.path.join(data_dir, f"shard_{idx:05d}.parquet") for idx in range(num_partitions)]
    return parquet_paths


def parquets_iter_batched(split, num_partitions=2, start=0, step=1):
    # start and step used in distributed training
    local_cache_dir = os.environ['NANOCHAT_BASEDIR']
    data_dir = local_cache_dir + '/base_data/'
    parquet_paths = list_parquet_files(data_dir, num_partitions)
    parquet_paths = parquet_paths[:-1] if split == 'train' else parquet_paths[-1:]
    for file in parquet_paths:
        pf = pq.ParquetFile(file)
        row_groups = pf.num_row_groups
        for row_idx in range(start, row_groups, step):
            rg = pf.read_row_group(row_idx)
            texts = rg.column('text').to_pylist()
            yield texts


def tokenize_text(tokenizer_from_tt, text):
    return tokenizer_from_tt.encode(text)


def tokenize_in_parallel(tokenizer_from_tt, text_list, max_workers=4) -> List[List[int]]:
    """Tokenize a list of texts in parallel using ThreadPoolExecutor."""
    tokenize_text_with_tokenizer = partial(tokenize_text, tokenizer_from_tt)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(tokenize_text_with_tokenizer, text_list)
        tokenized_results = list(results_iterator)
    return tokenized_results


def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda",
                                       file_num_partitions=2):
    # TODO: support distributed training by sharding the data loading
    # first get the document batch, returned are a chunck of texts
    def get_document_batch():
        # non stopping loop
        while True:
            # TODO: handle the distributed training
            for texts_chunk in parquets_iter_batched(split, num_partitions=file_num_partitions):
                # each texts_chunk is 1024 row, can use smaller, e.g. 128 rows
                for start_idx in range(0, len(texts_chunk), tokenizer_batch_size):
                    yield texts_chunk[start_idx: start_idx + tokenizer_batch_size]

    text_chunks_batch = get_document_batch()  # this is an iterator, and can iterate forever
    total_tokens_needed = B * T + 1
    print(total_tokens_needed)

    tokenizer = tiktoken.get_encoding("o200k_base")
    print(f"eot_token, {tokenizer.eot_token}")
    token_buffers = deque()

    # the following loop will iterate forever
    for texts_rows in text_chunks_batch:
        # texts_chunk are a list of texts
        # the following is the single thread version
        # for texts in texts_rows:
        #     tokenized = tokenizer.encode(texts)
        #     token_buffers.append(tokenizer.eot_token)
        #     token_buffers.extend(tokenized)

        tokenized_list = tokenize_in_parallel(tokenizer, texts_rows, tokenizer_threads)

        # TODO: find the answer: when to add the bos_token? (for tiktoken there is no bos_token, so using eot_token)
        token_buffers.append(tokenizer.eot_token)
        # this is in the pre-training stage, so no padding added at the end of the text chunk
        for tokenized in tokenized_list:
            token_buffers.extend(tokenized)

        while len(token_buffers) >= total_tokens_needed:
            encoded = [token_buffers.popleft() for _ in range(total_tokens_needed)]
            scratch_tensor = torch.tensor(encoded, dtype=torch.int64, pin_memory=(device == "cuda"))
            inputs_cpu = scratch_tensor[:-1].to(dtype=torch.int32)
            targets_cpu = scratch_tensor[1:]
            inputs = inputs_cpu.view(B, T).to(device=device, dtype=torch.int32)
            targets = targets_cpu.view(B, T).to(device=device, dtype=torch.int64)
            yield inputs, targets
