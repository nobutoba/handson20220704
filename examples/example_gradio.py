import gradio
import hydra
import torch
from my_package.utils.logger import get_logger
from my_package.utils.module_utils import instantiate
from omegaconf import DictConfig

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="default_demo.yaml")
def main(config: DictConfig):

    logger.info(f"Instantiating inference api <{config.inference_api._target_}>")
    inference_api = instantiate(config.inference_api)

    ckpt_dic = torch.load(config.model_state_dict)
    # Lightningの辞書キーから不要な文字列を削除し置換
    state_dict = {
        ".".join(key.split(".")[1:]): value
        for key, value in ckpt_dic["state_dict"].items()
    }
    inference_api.model.eval()
    inference_api.model.load_state_dict(state_dict)

    inference_func = inference_api.inference

    gradio_inputs = []
    for gradio_input in config.gradio_inputs:
        input_ = (
            instantiate(gradio_input)
            if hasattr(gradio_input, "_target_")
            else gradio_input
        )
        gradio_inputs.append(input_)

    gradio_outputs = [
        instantiate(gradio_output) for gradio_output in config.gradio_outputs
    ]

    gradio_interface = gradio.Interface(
        fn=inference_func,
        inputs=gradio_inputs,
        outputs=gradio_outputs,
        live=config.live,
    )
    gradio_interface.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
