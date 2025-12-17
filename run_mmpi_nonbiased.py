import os
import re
import gc
import sys
import time
import tqdm
import json
import traceback
import argparse

import string
import unicodedata

import pandas as pd
import collections
from pathlib import Path
from typing import (Any, Dict, List, Optional)

import torch
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

# Path settings
sys.path.append(str(Path(__file__)))
os.chdir(Path(__file__).resolve().parent)
print(os.getcwd())

from huggingface_hub import login

access_token = "YOUR-ACCESS-TOKEN"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = access_token

sys.path.insert(1, 'PATH-TO-CODE')
from evaluation_suit.model import AutoCausalLM

ASSETS_PATH = Path('./assets').resolve()
OUTPUT_PATH = Path("OUTPUTS-PATH").resolve()

DEFAULT_GENERATION_PARAMS = {
    "do_sample": False,
    "max_new_tokens": 3,
    "early_stopping": True,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "remove_invalid_values": True,
    "use_cache": True,
    "no_repeat_ngram_size": 4,
    "length_penalty": 0.01,
    "num_return_sequences": 1
}


def output_post_processing(input_quote):
    out = input_quote.strip()
    out = out.split("</s>")[0]
    out = out.strip("\n").strip()
    return out

def is_answered(answer: str):
    if ("true" in answer.lower()) or ("false" in answer.lower()): 
        return True
    elif ("yes" in answer.lower()) or ("no" in answer.lower()): 
        return True
    else: 
        return False

PUNCT = {chr(i) for i in range(sys.maxunicode) 
         if unicodedata.category(chr(i)).startswith('P')}.union(string.punctuation)

CHARACTERS = r"[a-zA-Zа-яА-Я]"

def remove_punc(text: str, punktuation: Optional[List[str]] = []):
    return ''.join(ch for ch in text if ch not in punktuation)


def extract_answer(answer):
    answer = output_post_processing(answer)
    if not is_answered(answer):
        print(f"Final answer was not found!")
        return None
    answer = remove_punc(answer, PUNCT)
    if "true" in answer.lower():
        return True
    else:
        return False


class MMPIDataset(Dataset):
    def __init__(self, prompt: str,
                 questions: List[str],
                 tokenizer = None, 
                 role: Optional[str] = None):
        self.prompt = prompt
        self.questions = questions
        self.tokenizer = tokenizer
        self.role = role

        if (self.role is not None) and self.role.islower():
            self.role = self.role.replace(" i ", " I ").strip()
            self.role = ". ".join(map(lambda x:  x[0].capitalize() + x[1:], self.role.split(". ")))
        
        
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        question = question[question.index(".") + 2:]
        question = question.strip()
        if not question.endswith("."):
            question = question + "."
    
        request = self.prompt.format(PERSONALITY=self.role, QUESTION=question)
        if self.tokenizer is not None:
            request_encoded = self.tokenizer(request, return_tensors="pt")
            input_len = len(request_encoded['input_ids'][0])
            request_encoded['input_len'] = input_len
            return request_encoded
        else:
            return request



def run_test_mmpi(model, 
                  test_questions: List[str], 
                  prompt: str,
                  roles: List[str],
                  max_roles: Optional[int] = 100,
                  batch_size: Optional[int] = 30,
                  generation_params: Optional[Dict[str, Any]] = DEFAULT_GENERATION_PARAMS,
                  output_filename: str = None,  # in json format
                  seed: int = 42):
    set_seed(seed)
    test_results = collections.defaultdict(list)  # for each person a list of answers (int: List[bool])

    def collate_fn(examples):
        encoded_inputs = model.tokenizer(examples,
                                        padding=True,
                                        return_tensors="pt")
        
        return {
            "input_ids": encoded_inputs['input_ids'],
            "attention_mask": encoded_inputs['attention_mask'],
            "prompt_length": encoded_inputs['attention_mask'].sum(1)
        }

    start_time = time.time()
    max_roles = min(max_roles, len(roles))
    for role_i in tqdm.tqdm(range(max_roles)):
        role = roles[role_i]
        dataset = MMPIDataset(prompt=prompt, 
                              questions=test_questions,
                              role=role)
        dataloader = DataLoader(dataset, 
                                shuffle=False, 
                                batch_size=batch_size,
                                collate_fn=collate_fn)
        role_test_answers = []
        for batch_i, batch in enumerate(dataloader):
            with torch.no_grad():
                out = model._model_generate(
                    inputs=batch,
                    max_tokens=generation_params['max_new_tokens'],
                    stop=[model.eot_token]
                )
                
            generated_texts = model.tokenizer.batch_decode(out, skip_special_tokens=True)
            generated_answers = [extract_answer(answer) for answer in generated_texts]
            role_test_answers.extend(generated_answers)

        if len(role_test_answers) != len(test_questions):
            print(f"WARNING: For a role #{role_i} got only {len(role_test_answers)} answers for {len(test_questions)} test questions!")
        test_results[role_i] = role_test_answers

    end_time = time.time()
    print(f"Evaluation finished.")
    print("Total run time %.2f" % (end_time - start_time))

    # Save results
    if output_filename is not None:
        with open(output_filename, 'w') as f:
            json.dump(dict(test_results), f, ensure_ascii=True, indent=2)
        print(f"Predictions saved: {output_filename}")

    gc.collect()
    torch.cuda.empty_cache()

    return dict(test_results)



def main(args):
    # Questions MMPI
    with open(str(ASSETS_PATH / "questions_mmpi2.txt"), "r") as f:
        questions = f.readlines()

    # Roles
    persons_lowercased = pd.read_csv(str(ASSETS_PATH / "personality.csv"), index_col=0)
    
    prompt = '''If I was described in a few sentences, I would say that: {PERSONALITY}
    When we were reviewing my test result, where whether the statement "{QUESTION}" is true or false about me, I answered it is'''



    print(f"----"*10)
    start_time = time.time()
    model_name = args.model_name
    batch_size = args.batch_size

    print(f"Model: {model_name}")
    try:
        model_args = f'pretrained={model_name},' \
                     f'tokenizer={model_name},' \
                     f'dtype="float16",use_accelerate=true,add_special_tokens=false'
        
        device_str = 'cuda'
        lm_model = AutoCausalLM.create_from_arg_string(
                    model_args,
                    {
                        "batch_size": batch_size,
                        "device": device_str,
                    },
                )
    except Exception as e:
        print(f"ERROR: Error occurred during {model_name} model loading. Skipping...")
        print(traceback.format_exc())


    try:
        lm_name = re.sub('\.', '-', re.sub('/', '-', model_name))
        EXPERIMENT_FILENAME = f"mmpi_llm-{lm_name}_v0.json"
                
        output_directory = Path(args.output_directory) if len(args.output_directory) else OUTPUT_PATH
        EXP_OUTPUT_PATH = output_directory / f"{lm_name}"
        if not EXP_OUTPUT_PATH.exists():
            EXP_OUTPUT_PATH.mkdir()
            print(f'{EXP_OUTPUT_PATH.name} created.')
        
        output_filename = EXP_OUTPUT_PATH / EXPERIMENT_FILENAME
        if output_filename.exists():
            print(f"\n\nSkipped...")
            return 0
                    
        print("Will be saved to: ", output_filename)
        test_results = run_test_mmpi(lm_model, 
                                    questions, 
                                    prompt=prompt,
                                    roles=persons_lowercased.Persona.tolist()[:25],
                                    batch_size=batch_size,
                                    output_filename=output_filename)
    except Exception as e:
        print(f"ERROR: Error occurred during {model_name} model evaluation. Skipping...\n{e}")
        print(traceback.format_exc())

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n{duration} seconds taken to run inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--output_directory", type=str, default="")
    args = parser.parse_args()

    main(args)
    


