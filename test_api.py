from llm_interface import query_model

result = query_model("mistralai/mistral-7b-instruct", "x + x")
print("Model Output:", result)