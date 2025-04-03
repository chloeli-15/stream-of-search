from vllm import LLM, SamplingParams
from datasets import load_dataset

# model = OnlineLM("deepseek-ai/DeepSeek-R1")
data_all = load_dataset("MelinaLaimon/stream-of-search")
data = data_all["test"].select(range(10))

data = data.map(lambda x: { # type: ignore
    'test_prompt': [
        # {'role': 'system', 'content': SYSTEM_PROMPT},
        x["messages_sos"]["role"=="user"]
    ],
    # 'answer': extract_hash_answer(x['answer'])
})

sampling_params = SamplingParams(
    temperature=0.1,
    top_k=40,
    top_p=0.95,
)

llm = LLM(model="chloeli/qwen-2.5-0.5B-instruct-sft-lora-countdown-optimal-1k")
outputs = llm.generate(data['test_prompt'], sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    