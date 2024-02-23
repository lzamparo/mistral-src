import os
import torch
import pickle as pkl
from typing import List

from mistral.model import ModelArgs, Transformer
from mistral.moe import MoeArgs
from main import generate


class DebugTokenizer:
    @property
    def bos_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return 1

    @property
    def pad_id(self) -> int:
        return -1

    def encode(self, s: str, bos: bool = True) -> List[int]:
        assert isinstance(s, str)
        t = [int(x) for x in s.split()]
        if bos:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: List[int]) -> str:
        return " ".join([str(x) for x in t])


def test_generation():
    torch.manual_seed(42)
    temperature = 0.2
    sequences = ["1 2 3 4 5 6 7", "0 1 2", "12 13 14", "2 4 34"]
    args = ModelArgs(
        dim=64,
        n_layers=1,
        head_dim=64,
        hidden_dim=128,
        n_heads=2,
        n_kv_heads=2,
        sliding_window=3,
        norm_eps=1e-5,
        vocab_size=32_000,
        max_batch_size=len(sequences),
    )
    model = Transformer(args).to("cuda", dtype=torch.float32)
    tokenizer = DebugTokenizer()

    # for attempt in range(10):
    toks, all_logprobs_old = generate(sequences, model, tokenizer, temperature=temperature, max_tokens=7)
    toks = [" ".join(r.split(" ")[1:]) for r in toks] # Remove BOS
    generated, all_logprobs_new = generate(toks, model, tokenizer, temperature=temperature, max_tokens=0)
    assert generated == []
    
    # Verify that logprobs are the same
    assert len(sequences) == len(all_logprobs_old) == len(all_logprobs_new)
    for lp_old, lp_new in zip(all_logprobs_old, all_logprobs_new):
        assert all([abs(x - y) < 1e-5 for x, y in zip(lp_old, lp_new)]), f"\n{lp_old}\n{lp_new}"

    print("All tests passed.")


def test_save_output_and_tensors(outdir: str):
    torch.manual_seed(42)
    temperature = 0.2
    sequences = ["1 2 3 4 5 6 7", "0 1 2", "12 13 14", "2 4 34"]
    args = ModelArgs(
        dim=64,
        n_layers=2,
        head_dim=64,
        hidden_dim=128,
        n_heads=2,
        n_kv_heads=2,
        sliding_window=3,
        norm_eps=1e-5,
        vocab_size=32_000,
        max_batch_size=len(sequences),
        moe=MoeArgs(num_experts=2, num_experts_per_tok=1),
    )
    model = Transformer(args).to("cuda", dtype=torch.float32)
    tokenizer = DebugTokenizer()

    # for attempt in range(10):
    toks, all_logprobs = generate(sequences, model, tokenizer, temperature=temperature, max_tokens=7)

    # save tensors: toks, all_logprobs, sequences, tokenizer
    with open(os.path.join(outdir, "generated_tokens.pkl"),'wb') as f:
        pkl.dump(toks, f)
    with open(os.path.join(outdir, "generated_logprobs.pkl"),'wb') as f:
        pkl.dump(all_logprobs, f)
    with open(os.path.join(outdir, "seqs_and_tokenizer.pkl"),'wb') as f:
        pkl.dump((sequences, tokenizer), f)

    # model to text
    model_cpu = model.to("cpu")
    with open(os.path.join(outdir, "model_tensors.txt"),'w') as f:
        for param_tensor in model_cpu.state_dict():
            print(param_tensor, "\t", model_cpu.state_dict()[param_tensor].size(), file=f)
        
    # save model dict
    torch.save(model_cpu.state_dict(), os.path.join(outdir, "model_sd.pt"))


def test_chunks():
    torch.manual_seed(42)

    sequences = [" ".join([str(i) for i in range(7)]), " ".join([str(i) for i in range(9, 0, -1)])]
    args = ModelArgs(
        dim=512,
        n_layers=1,
        head_dim=128,
        hidden_dim=2048,
        n_heads=4,
        n_kv_heads=2,
        sliding_window=4,
        norm_eps=1e-5,
        vocab_size=32_000,
        max_batch_size=3,
    )
    model = Transformer(args).to("cuda", dtype=torch.float32)
    tokenizer = DebugTokenizer()

    # for attempt in range(10):
    toks, all_logprobs_old = generate(sequences, model, tokenizer, max_tokens=8)
    toks = [" ".join(r.split(" ")[1:]) for r in toks] # Remove BOS
    generated, all_logprobs_new = generate(toks, model, tokenizer, max_tokens=0, chunk_size=5)
    assert len(generated) == 0

    for lp_old, lp_new in zip(all_logprobs_old, all_logprobs_new):
        assert all([abs(x - y) < 1e-5 for x, y in zip(lp_old, lp_new)]), f"\n{lp_old}\n{lp_new}"
    

if __name__ == "__main__":
    outdir = os.path.expanduser("~/mixtral_weights/moe_2_small")
    #test_generation()
    test_save_output_and_tensors(outdir)
    # test_chunks()
