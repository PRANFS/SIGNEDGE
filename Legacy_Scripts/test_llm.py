from llama_cpp import Llama

print("--- Loading Model (this takes ~5-10 seconds) ---")
# We use 1 thread to respect your 2-core rule for now
llm = Llama(model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", n_ctx=512, n_threads=1)

print("--- Sending Prompt ---")
output = llm("The capital of France is", max_tokens=10, echo=True)

print("\n--- Result ---")
print(output['choices'][0]['text'])