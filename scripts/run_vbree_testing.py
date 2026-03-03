from dotenv import load_dotenv

from vbree.orchestration.ensemble import Ensemble
from vbree.providers.hf_provider import HfProvider

from datasets import load_dataset

def main():
    load_dotenv()
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split = "validation"). to_pandas()
    
    df_test = dataset.head(1)

    df_test["options"] = df_test["options"].apply(list)

    print(type(df_test["options"][0]))

    # define providers and models
    providers = {
        "Qwen/Qwen2.5-7B-Instruct:together": HfProvider("Qwen/Qwen2.5-7B-Instruct:together"),
        "meta-llama/Llama-3.1-8B-Instruct:cerebras": HfProvider("meta-llama/Llama-3.1-8B-Instruct:cerebras"),
        "openai/gpt-oss-20b": HfProvider("openai/gpt-oss-20b")
    }

    ensemble = Ensemble(providers=providers, verbose=True)

    ensemble.add_model("Qwen/Qwen2.5-7B-Instruct:together")
    ensemble.add_model("meta-llama/Llama-3.1-8B-Instruct:cerebras")
    ensemble.add_model("openai/gpt-oss-20b")


    results =ensemble.run(df_test,id_col="question_id",question_col="question", choices_col="options", domain_col="category")
    
    results.to_csv("runs/v_bree_test_module.csv", index=False)

if __name__ == "__main__":
    main()






