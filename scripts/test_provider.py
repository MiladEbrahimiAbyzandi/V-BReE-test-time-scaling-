from vbree.providers.hf_provider import HfProvider

provider= HfProvider(model="Qwen/Qwen2.5-7B-Instruct")


response = provider.generate("What is the capital of France?")

print (response)
