import chz
import sys
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
import asyncio

MODEL="moonshotai/Kimi-K2-Thinking"
DATASET="data/data_new.jsonl"
LOGS_DIR="logs/Kimi-K2_New/"

def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = MODEL 
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=32768,
        batch_size=64,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config, file_path=DATASET
    )
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": LOGS_DIR,
            "model_name": model_name,
            "dataset_builder": dataset,
            "learning_rate": 2e-4,
            "lr_schedule": "linear",
            "lora_rank": 32,
            "num_epochs": 1,
        }
    )

def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))

if __name__ == "__main__":
    load_dotenv()
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
