from tqdm import tqdm
import torch
from torch.nn.functional import pad

from transformers import GPT2Tokenizer
from smoothquant.opt import Int8OPTForCausalLM

class Evaluator:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0
        for batch in tqdm(self.dataset):
            input_ids = batch['input_ids'].cuda().unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            outputs = model(input_ids)
            end.record()
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        acc = hit / total
        lantecy = latency / len(self.dataset)
        return acc, lantecy
    

def print_model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('>>>>>>>>>>>>>> Model size: {:.3f}MB'.format(size_all_mb))


from datasets import load_dataset
tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-2.7b')
dataset = load_dataset('lambada', split='validation[:1000]')
evaluator = Evaluator(dataset, tokenizer)

model_smoothquant = Int8OPTForCausalLM.from_pretrained(
    'mit-han-lab/opt-2.7b-smoothquant', torch_dtype=torch.float16, device_map='cuda:0')
print_model_size(model_smoothquant)
acc_smoothquant, lantecy_smoothquant = evaluator.evaluate(model_smoothquant)
print(
    f'SmoothQuant INT8 accuracy: {acc_smoothquant}, per-sample lantecy: {lantecy_smoothquant:.3f}ms')