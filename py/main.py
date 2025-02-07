from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer

model_dir = "./models/chat"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
token_ids = tokenizer.encode("Once upon a time")
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)
# 文本嵌入
# 导入llama模型
config = AutoConfig.from_pretrained(model_dir)
print(config)
# 加载模型
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
model = LlamaForCausalLM.from_pretrained(model_dir, config=config)
generate_ids = model.generate(inputs.input_ids, max_length=100)
tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
