import json
import random

from loguru import logger
from tqdm import tqdm
import torch

from models.baseline_generation_model import GenerationModel
import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, \
    BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import torch

class Qwen3Model(GenerationModel):
    def __init__(self):

       # super().__init__(config=config)
        llm_path = "Qwen/Qwen3-8B"
        use_quantization = False
        self.few_shot = 5
        self.batch_size = 4
        self.max_new_tokens = 64
        #self.stop_strings= ["\n"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            padding_side="left",
        )

        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                device_map="auto"
            )
        self.pipe = pipeline(
            task="text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
        )
        self.system_message = (
            "Given a question, your task is to provide the list of answers without any other context. "
            "If there are multiple answers, separate them with a comma. "
            "If there are no answers, type \"None\".")

    def instantiate_in_context_examples(self, train_data_file):
        logger.info(f"Reading train data from `{train_data_file}`...")
        with open(train_data_file) as f:
            train_data = [json.loads(line) for line in f]

        # instantiate templates with train data
        logger.info("Instantiating in-context examples with train data...")

        in_context_examples = []
        for row in train_data:
            template = self.prompt_templates[row["Relation"]]
            example = {
                "relation": row["Relation"],
                "messages": [
                    {
                        "role": "user",
                        "content": template.format(subject_entity=row["SubjectEntity"])
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f'{", ".join(row["ObjectEntities"]) if row["ObjectEntities"] else "None"}')
                    }
                ]
            }

            in_context_examples.append(example)

        return in_context_examples

    def create_prompt(self, subject_entity: str, relation: str) -> str:
        template = self.prompt_templates[relation]
        random_examples = []
        if self.few_shot > 0:
            pool = [
                example["messages"] for example in self.in_context_examples if example["relation"] == relation
            ]
            random_examples = random.sample(
                pool,
                min(self.few_shot, len(pool))
            )

        messages = [
            {
                "role": "system",
                "content": self.system_message
            }
        ]

        for example in random_examples:
            messages.extend(example)

        messages.append({
            "role": "user",
            "content": template.format(subject_entity=subject_entity)
        })

        prompt = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_predictions(self, inputs, infer_technique):
        logger.info("Generating predictions...")
    #https://huggingface.co/docs/transformers/generation_strategies
        if infer_technique == 'greedy':
            output = self.pipe(
                            inputs,
                            max_new_tokens=self.max_new_tokens,
                            return_full_text=False,
                            do_sample=False
                        )

        elif infer_technique == 'sampling':
            output = self.pipe(
                            inputs,
                            max_new_tokens=self.max_new_tokens,
                            return_full_text=False,
                            do_sample=True, 
                            num_beams=1,
                        
                        )

        elif infer_technique == 'beam':
            output = self.pipe(
                            inputs,
                            max_new_tokens=self.max_new_tokens,
                            return_full_text=False,
                            num_beams=2,
                            use_cache=False 
                        )
        

        elif infer_technique == 'dola':
            output = self.pipe(
                            inputs,
                            max_new_tokens=self.max_new_tokens,
                            return_full_text=False,
                            do_sample=False,
                            dola_layers="high"
                        )
        else:
            output = "ERROR " + infer_technique
            print(output)


            
        
        

        return output
