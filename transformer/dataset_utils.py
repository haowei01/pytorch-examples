import torch
import torch.distributions as dist
from torch.utils.data import DataLoader, IterableDataset
import datasets
from transformers import AutoTokenizer

tokenizer_opus_de_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

dataset_wmt14_de_en = datasets.load_dataset("wmt14", "de-en", streaming=True)


def preprocess_function(examples, max_length=128, tokenizer=tokenizer_opus_de_en):
    translations = examples["translation"]
    batch_size = len(translations)
    inputs = [ex['de'] for ex in translations]
    outputs = [ex['en'] for ex in translations]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")

    labels = tokenizer(outputs, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    # teacher forcing
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    bos = torch.zeros(batch_size, 1, dtype=torch.long)
    if not bos_token_id:
        bos.fill_(bos_token_id)
    model_inputs["teacher_forcing_input_ids"] = torch.cat([bos, labels["input_ids"][:, :-1]], dim=1)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_dataloader(
    dataset, batch_size=8, train_eval='train', max_length=128, tokenizer=tokenizer_opus_de_en, num_workers=1
):
    processed_dataset = dataset.map(preprocess_function, batched=True)
    processed_dataset = processed_dataset[train_eval]
    dataloader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=num_workers,
    )
    return dataloader


class DistributedIterableDatasetWrapper(IterableDataset):
    """
    Wrapper to make streaming datasets work with DDP by implementing proper sharding
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        """
        Create iterator that yields different samples for each GPU process
        """
        # Create iterator from the streaming dataset
        iterator = iter(self.dataset)

        # Skip samples to ensure each process gets different data
        # Process 0 gets samples 0, num_replicas, 2*num_replicas, ...
        # Process 1 gets samples 1, num_replicas+1, 2*num_replicas+1, ...
        sample_idx = 0

        for sample in iterator:
            # Only yield samples assigned to this process (rank)
            if sample_idx % self.num_replicas == self.rank:
                yield sample
            sample_idx += 1

    def set_epoch(self, epoch):
        """Set epoch for shuffling (if implemented)"""
        self.epoch = epoch


def prepare_distributed_iterable_dataloader(
    dataset, batch_size=8, train_eval='train', max_length=128, tokenizer=tokenizer_opus_de_en, num_workers=1,
    rank=None, num_replicas=None
):
    processed_dataset = dataset.map(preprocess_function, batched=True)
    processed_dataset = processed_dataset[train_eval]
    distributed_iterable_dataset = DistributedIterableDatasetWrapper(processed_dataset, rank=rank, num_replicas=num_replicas)
    dataloader = DataLoader(
        distributed_iterable_dataset,
        batch_size=batch_size,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=num_workers,
    )
    return dataloader



