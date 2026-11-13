"""Mock inference is synthetic. Transformers is imported only for explicit GPU runs."""

import hashlib


class MockModel:
    synthetic = True
    metadata = {
        "backend": "mock",
        "synthetic": True,
        "purpose": "pipeline validation only",
    }

    def generate(self, case, condition):
        if condition["condition"] == "injection" and case["kind"] == "attack":
            text = case["marker"]
        else:
            text = case["aliases"][0]
        return dict(
            raw_output=text, error=None, truncated=False, prompt=condition["messages"]
        )


def render_chat(tokenizer, messages, mode="auto"):
    if mode not in ("auto", "system", "fold"):
        raise ValueError("system_role must be auto, system or fold")
    if not tokenizer.chat_template:
        raise ValueError(
            "Model has no chat template; review model card before inference"
        )
    if mode != "fold":
        # Check this actual tokenizer/template, not the model family's assumed role support.
        probe = [
            dict(role="system", content="LEXSI_SYSTEM_PROBE"),
            dict(role="user", content="LEXSI_USER_PROBE"),
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=True
            )
            if not (
                "LEXSI_SYSTEM_PROBE" in rendered
                and "LEXSI_USER_PROBE" in rendered
                and rendered.index("LEXSI_SYSTEM_PROBE")
                < rendered.index("LEXSI_USER_PROBE")
            ):
                raise ValueError("Template discards or reorders system text")
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ), "system"
        except Exception as exc:
            # Template role rejection commonly raises jinja2.TemplateError.
            # An explicit system request must fail, while auto may use recorded folding.
            if (
                not isinstance(exc, ValueError)
                and type(exc).__name__ != "TemplateError"
            ):
                raise
            if mode == "system":
                raise ValueError(
                    "Actual template does not accept system messages"
                ) from exc
    folded = [
        dict(
            role="user",
            content="Assistant instruction:\n"
            + messages[0]["content"]
            + "\n\n"
            + messages[1]["content"],
        )
    ]
    return tokenizer.apply_chat_template(
        folded, tokenize=False, add_generation_prompt=True
    ), "fold"


class TransformersModel:
    synthetic = False

    def __init__(self, config):
        import torch
        from huggingface_hub import model_info
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            set_seed,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Real model runner requires a GPU; use --model mock for CPU validation"
            )
        set_seed(config["seed"])
        self.config = config
        # Authentication comes from HF_TOKEN / huggingface_hub's secure credential cache.
        sha = model_info(config["model_id"], revision=config["revision"]).sha
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], revision=sha, trust_remote_code=False
        )
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        options = dict(
            revision=sha, device_map="auto", dtype=self.dtype, trust_remote_code=False
        )
        if config["load_in_4bit"]:
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_id"], **options
        ).eval()
        _, mode = render_chat(
            self.tokenizer,
            [
                dict(role="system", content="Be helpful."),
                dict(role="user", content="Hello"),
            ],
            config["system_role"],
        )
        self.metadata = dict(
            backend="transformers",
            synthetic=False,
            model_id=config["model_id"],
            revision=sha,
            role_mode=mode,
            chat_template=self.tokenizer.chat_template,
            chat_template_sha256=hashlib.sha256(
                str(self.tokenizer.chat_template).encode()
            ).hexdigest(),
            gpu=torch.cuda.get_device_name(),
            dtype=str(self.dtype),
            config=config,
            model_type=self.model.config.model_type,
            device_map=str(self.model.hf_device_map),
            do_sample=False,
            num_beams=1,
            quantization="nf4" if config["load_in_4bit"] else None,
        )

    def generate(self, case, condition):
        import torch

        rendered, mode = render_chat(
            self.tokenizer, condition["messages"], self.config["system_role"]
        )
        tokens = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        length = tokens["input_ids"].shape[-1]
        # Never truncate factual content or silently drop the attack from a matched condition.
        if length > self.config["max_input_tokens"]:
            return dict(
                raw_output="",
                error=f"input_too_long:{length}",
                truncated=False,
                prompt=rendered,
            )
        inputs = {k: v.to(self.model.device) for k, v in tokens.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.config["max_new_tokens"],
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            )
        generated = output[0, length:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else (eos or [])
        truncated = (
            len(generated) >= self.config["max_new_tokens"]
            and int(generated[-1]) not in eos_ids
        )
        return dict(
            raw_output=self.tokenizer.decode(generated, skip_special_tokens=True),
            error=None,
            truncated=truncated,
            prompt=rendered,
            input_tokens=length,
            output_tokens=len(generated),
            role_mode=mode,
        )
