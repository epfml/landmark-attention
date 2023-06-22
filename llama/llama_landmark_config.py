from transformers.models.llama.configuration_llama import LlamaConfig

class LlamaLandmarkConfig(LlamaConfig):
    model_type = "llama_with_landmark"

    def __init__(
        self,
        mem_id=32001,
        mem_freq=50,
        train_context_length=512,
        include_landmark_in_loss=True,
        **kwargs,
    ):
        self.mem_id = mem_id
        self.mem_freq = mem_freq
        self.train_context_length = train_context_length
        self.include_landmark_in_loss = include_landmark_in_loss
        super().__init__(**kwargs)
