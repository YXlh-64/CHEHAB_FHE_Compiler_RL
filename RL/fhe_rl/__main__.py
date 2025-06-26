from .run import run_agent
from .train import train_agent
from .test import test_agent
import sys
from .utils import load_embedding_model

if __name__ == "__main__":
    embeddings_model = load_embedding_model(
        checkpoint_path="./trained_models/model_Transformer_ddp_10399047_epoch_5000000.pth",
        device="cpu"
    )
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|test] [model_filepath (for test)]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    if mode == "train":
        train_agent("./all_expressions_cleaned.txt",embeddings_model, total_timesteps=1000000)
    elif mode == "test":
        if len(sys.argv) < 3:
            print("Usage: python main.py test [model_filepath]")
            sys.exit(1)
        model_filepath = sys.argv[2]
        test_agent("./test_expressions_merged.txt",embeddings_model, model_filepath)
    elif mode == "run":
        if len(sys.argv) < 5:
            print("Usage: python main.py run [model_filepath] [input_file] [output_file]")
            sys.exit(1)
        model_filepath = sys.argv[2]
        input_file = sys.argv[3]
        output_file = sys.argv[4]
        run_agent(input_file,embeddings_model, model_filepath,output_file)
    else:
        print("Invalid command. Use 'train' or 'test' or 'run'.")