import warnings
warnings.filterwarnings("ignore")

from langchain.llms import HuggingFacePipeline

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline

class LLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_config = {
            'llama': {
                'tokenizer': 'aleksickx/llama-7b-hf',
                'model': 'aleksickx/llama-7b-hf',
                'T': 0.1
            },
            'bloom': {
                'tokenizer': 'bigscience/bloom-7b1',
                'model': 'bigscience/bloom-7b1',
                'T': 0
            },
            'falcon': {
                'tokenizer': 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2',
                'model': 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2',
                'T': 0
            }
        }

    def get_model(self):
        if self.model_name not in self.model_config:
            print("Given model is not available!")
            return None, None, None, None, None

        config = self.model_config[self.model_name]
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
        model = AutoModelForCausalLM.from_pretrained(
            config['model'],
            load_in_8bit=True,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        max_len = 1024
        task = "text-generation"
        T = config['T']

        return tokenizer, model, max_len, task, T

    def get_pipeline(self):
        tokenizer, model, max_len, task, T = self.get_model()
        if tokenizer is None or model is None:
            return None

        pipe = pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            max_length=max_len,
            temperature=T,
            top_p=0.95,
            repetition_penalty=1.15
        )

        return HuggingFacePipeline(pipeline=pipe)