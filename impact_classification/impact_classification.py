import pandas as pd
data = pd.read_csv('../../dataset.csv')
#data['MAIN_CAT'] = data['MAIN_CAT'].apply(lambda x: 'ANOIMPACT' if x == 'ACADEMIC' else x)
data['label'] = data['MAIN_CAT'].apply(lambda x: 0 if x == 'ANOIMPACT' else 1)
data.groupby('label').count()
#data['label'] = data['MAIN_CAT'].apply(lambda x: 0 if x == 'ACADEMIC' else x)
data['len'] = data['MAIN_CAT'].apply(lambda x:len(x.split(',')))
data = data[data['len']==1]
data['MAIN_CAT'].unique()
data = data.sample(frac=1, random_state = 1)
data.index = range(len(data))


# In[2]:



# In[ ]:


import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset


base_model = "meta-llama/Llama-2-7b-hf/Llama-2-7b-chat-hf"
# Fine-tuned model name
new_model = "llama-2-7b-chat-fine-tune"

# Prepare dataa
ddd = data[data['MAIN_CAT'] != 'ANOIMPACT']
#ddd = pd.concat([data[data['MAIN_CAT'] == 'ANOIMPACT'].sample(n=len(dd)*2//5, random_state = 1),dd])
ddd['MAIN_CAT'] = ddd['MAIN_CAT'].str.replace('ANOIMPACT', 'NO-IMPACT').str.replace('POLITICALLEGISLATIVE', 'POLITICAL-LEGISLATIVE').str.replace('AOTHERMain', 'OTHER')
ddd['SUB_CAT'] = ddd['SUB_CAT'].apply(lambda x: x.split('.')[1])
ddd.groupby(['SUB_CAT']).count()


#subcategory
#ddd['text'] = "<s>[INST] " + "<<SYS>> You will be provided with a text that possibly outlines the impact of a research project. We define impact as an effect of scientific activities within academia or beyond the academic field, e.g. on the scientific field, economy, society, culture, politics, law, technology or the environment. In a report, impact may be represented by describing methods and routines implemented for a project, or by the impact that authors anticipated when writing their final reports. This means that estimated impact is also considered as impact. Impact could also be the maintenance or the avoidance of change. Your task is to assign one impact category to the text from the following options: ['No Impact', 'Data Policy', 'Future Research', 'OTHER', 'Income', 'Model/algorithm development', 'Research Methods', 'Learning and Teaching', 'Knowledge Transfer', 'Academic Knowledge Acquisition', 'Regulations', 'IT security', 'PR/Visibility', 'Collaborations', 'Prototype', 'Publications', 'Knowledge Acquisition', 'Product Development/Improvement', 'Income Academia', 'Education', 'Business Models', 'Documentation', 'Awareness', 'Climate Protection', 'Physical Health', 'Optimizing Processes', 'Culture/Events', 'Academic Events', 'Culture', 'Employee Satisfaction', 'Data Collection/Release', 'Life Quality', 'City Facility', 'Sustainability', 'Laws', 'Safety', 'Justice', 'Mobility'] <</SYS>> TEXT: " + ddd['TEXT'] + ' [/INST] Main Category: ' + ddd['MAIN_CAT'] + '; Sub Category: ' + ddd['SUB_CAT'] +  '. </s>'

#main category
promp = """You will be provided with a text that possibly outlines the impact of a research project. We define impact as an effect of scientific activities within academia or beyond the academic field, e.g. on the scientific field, economy, society, culture, politics, law, technology or the environment. In a report, impact may be represented by describing methods and routines implemented for a project, or by the impact that authors anticipated when writing their final reports. This means that estimated impact is also considered as impact. Impact could also be the maintenance or the avoidance of change.

Your task is to assign one impact category to the text from the following options: 

NO-IMPACT: the sentence does not express any impact.
TECHNICAL IMPACT: refers to technologies that are used outside of the original project, e.g. software prototype development, Improving IT security, or data release
ECONOMIC IMPACT: refers to the use of research results for economic developments, e.g. development of business models, service quality, or economic strategies
ACADEMIC IMPACT: refers to impact within academia - within or beyond the own field/institution, e.g.improved learning and teaching, new research methods, or publications
SOCIETAL IMPACT: occurs when a project influences societal groups or institutions like schools, local authorities, foundations, or clubs, refugees/migration, religious persecution, etc.
POLITICAL-LEGISLATIVE IMPACT: refers to using the project results in political or legislative contexts, e.g. contributions to laws or political regulations
ETHICAL IMPACT: refers to ethical impact, e.g. equality, awareness, or charity
ENVIRONMENTAL IMPACT: refers to changes of ecological or environmental aspects, e.g. climate protection, protection of species, or sustainability of products
OTHER: any sentences that express impact, which are not captured by the previous categories.

Simply return one category without any explanation."""

ddd['text'] = "<s>[INST] <<SYS>>\n" + promp + "\n<</SYS>>\n\nText: " + ddd['TEXT'] + ' Category: [/INST] ' + ddd['MAIN_CAT'] + '.</s>'


dd_input = ddd[ddd['TRAIN_TEST'] == 'TRAIN']


val = dd_input.sample(frac = 0.15, random_state = 10)
tra = dd_input[~dd_input.index.isin(val.index)]
#tra = dd_input

# resample and balance classes
min_size = tra[~tra['MAIN_CAT'].isin(['ENVIRONMENTAL', 'ETHICAL', 'POLITICAL-LEGISLATIVE'])]['MAIN_CAT'].value_counts().min()

lst = []
for class_index, group in tra[tra['MAIN_CAT'].isin(['ENVIRONMENTAL', 'ETHICAL', 'POLITICAL-LEGISLATIVE'])].groupby('MAIN_CAT'):
    lst.append(group.sample(min_size, replace=True, random_state = 1))
frame_new = pd.concat(lst)

tra[~tra['MAIN_CAT'].isin(['ENVIRONMENTAL', 'ETHICAL', 'POLITICAL-LEGISLATIVE'])]
tra = pd.concat([tra[~tra['MAIN_CAT'].isin(['ENVIRONMENTAL', 'ETHICAL', 'POLITICAL-LEGISLATIVE'])], frame_new])
tra = tra.sample(frac = 1, random_state = 1)


from datasets import Dataset
tra_input = Dataset.from_pandas(tra[['text']], split='train')
val_input = Dataset.from_pandas(val[['text']], split='validation')
tra_input = tra_input.remove_columns('__index_level_0__')
val_input = val_input.remove_columns('__index_level_0__')


# fine-tuning
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map = 'auto'
)
model.config.use_cache = False
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"




peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
)



training_params = TrainingArguments(
    output_dir="./llama2_7_main",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_total_limit=2,
    save_steps=600,
    logging_steps=600,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    evaluation_strategy="steps"
)



trainer = SFTTrainer(
    model=model,
    train_dataset=tra_input,
    eval_dataset=val_input,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)





trainer.train()




trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)


