import os
import sys
import torch
import warnings
import torch.distributed as dist
from datasets import load_from_disk
from transformers import (
    set_seed,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from models.personal_model import DEPModel

warnings.filterwarnings("ignore")
set_seed(42)

class CustomTrainer(Seq2SeqTrainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"output/checkpoint-{self.state.global_step}"
        os.makedirs(checkpoint_folder, exist_ok=True)

        target_prefixes = [ 
            "align_mlp_user", 
            "align_mlp_friend", 
            "align_mlp_graph", 
            "gcn",

        ]

        base_model = model.module if hasattr(model, "module") else model
        selected_state_dict = {
            name: param.detach().cpu()
            for name, param in base_model.named_parameters()
            if any(name.startswith(prefix) or f".{prefix}." in name for prefix in target_prefixes)
        }

        torch.save(selected_state_dict, os.path.join(checkpoint_folder, "trainable_modules.pt"))
        self.tokenizer.save_pretrained(checkpoint_folder)

        print(f"Saved {len(selected_state_dict)} trainable params to {checkpoint_folder}/trainable_modules.pt")
    
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )
    
def custom_collate_fn(batch):
    collated = {}


    collated["graph_data"] = [b.get("graph_data", None) for b in batch]


    if "embeddings" in batch[0]:
        emb_list = []
        for b in batch:
            emb = b["embeddings"]
            if isinstance(emb, list):
                emb = torch.tensor(emb, dtype=torch.float32)
            elif isinstance(emb, torch.Tensor):
                emb = emb.to(torch.float32)
            else:
                raise TypeError(f"embeddings type not supported: {type(emb)}")
            emb_list.append(emb)
        collated["embeddings"] = torch.stack(emb_list, dim=0)

    keys = set().union(*[b.keys() for b in batch])
    keys.discard("graph_data")
    keys.discard("embeddings")
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            collated[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            collated[k] = [b[k] for b in batch]

    return collated

llm_model_name = "Qwen/Qwen2.5-7B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained("data/yelp_tokenizer")

if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

personal_model = DEPModel.from_pretrained(
    llm_model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    training=True,
    tokenizer=llm_tokenizer,
    cache_dir="/root/autodl-tmp/my_project/model_point",
)
personal_model.resize_token_embeddings(len(llm_tokenizer))

print_trainable_parameters(personal_model)

train_dataset = load_from_disk("data/processed_train")

training_args = Seq2SeqTrainingArguments(
    num_train_epochs=5,
    output_dir="output",
    logging_steps=10,
    save_strategy="epoch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="adamw_torch",
    learning_rate=1e-5,
    weight_decay=0.025,
    warmup_ratio=0.01,
    bf16=True,
    deepspeed="deepspeed/ds_z1_config.json",
    report_to="wandb",
    run_name="cold-start",
    remove_unused_columns=False,
    metric_for_best_model="loss",
    greater_is_better=False,
    dataloader_num_workers=0,
    fp16=False,
    ddp_find_unused_parameters=False,
)

trainer = CustomTrainer(
    model=personal_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=llm_tokenizer,
    data_collator=custom_collate_fn,
)

print("Training start...\n")
try:
    trainer.train()
    print("Training finished.\n")
except Exception as e:
    print(f" Training failed: {e}")
    import traceback
    traceback.print_exc()

if dist.is_initialized():
    dist.destroy_process_group()


sys.exit(0)

