"""Offline CPU proof of the real VLM collator, loss masks, and backward pass.

This small randomly initialized LLaVA is a plumbing test, not a useful CAD model.
No model weights or tokenizer are downloaded.
"""

import argparse
import json
from pathlib import Path


def vision_smoke(dataset_root, output_dir):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
        LlavaProcessor,
        PreTrainedTokenizerFast,
    )
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    from .train import check_vision_supervision, load_export

    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new smoke directory")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    dataset = load_export(Path(dataset_root) / "exports/train.feedback.jsonl")
    examples = [dataset[i] for i in range(min(4, len(dataset)))]
    special = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "<image>",
        "<system>",
        "<user>",
        "<assistant>",
    ]
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [
            json.dumps({"prompt": e["prompt"], "completion": e["completion"]})
            for e in examples
        ],
        trainers.BpeTrainer(vocab_size=512, special_tokens=special),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=special[4:],
    )
    template = "{% for message in messages %}{{ '<' + message['role'] + '>' }}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for block in message['content'] %}{% if block['type'] == 'image' %}{{ '<image>' }}{% else %}{{ block['text'] }}{% endif %}{% endfor %}{% endif %}{{ '<eos>' }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    tokenizer.chat_template = template
    processor = LlavaProcessor(
        image_processor=CLIPImageProcessor(
            size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28}
        ),
        tokenizer=tokenizer,
        patch_size=14,
        vision_feature_select_strategy="default",
        num_additional_image_tokens=1,
        chat_template=template,
    )
    config = LlavaConfig(
        vision_config=CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=28,
            patch_size=14,
        ),
        text_config=LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=4096,
        ),
        image_token_index=tokenizer.convert_tokens_to_ids("<image>"),
        image_seq_length=4,
        vision_feature_layer=-1,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlavaForConditionalGeneration(config)
    collator = DataCollatorForVisionLanguageModeling(
        processor=processor, completion_only_loss=True, max_length=None
    )
    supervision = check_vision_supervision(examples, processor, 4096, 1)
    batch = collator(examples[:2])
    image_mask = batch["input_ids"] == config.image_token_index
    if not image_mask.any() or not (batch["labels"][image_mask] == -100).all():
        raise AssertionError("Image tokens must be present and masked from target loss")
    before = model.get_output_embeddings().weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    output = model(**batch)
    if not torch.isfinite(output.loss):
        raise AssertionError("VLM loss is not finite")
    output.loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    delta = float((before - model.get_output_embeddings().weight.detach()).abs().max())
    if norm <= 0 or delta <= 0:
        raise AssertionError("VLM optimization did not update weights")
    report = {
        "passed": True,
        "device": "cpu",
        "model": "random tiny LLaVA, offline",
        "parameters": model.num_parameters(),
        "batch_rows": len(examples[:2]),
        "image_tokens": int(image_mask.sum()),
        "all_image_tokens_masked": True,
        "supervised_tokens": int((batch["labels"] != -100).sum()),
        "loss": float(output.loss.detach()),
        "grad_norm": norm,
        "max_weight_delta": delta,
        "supervision": supervision,
        "claim": "Real multimodal forward/backward and masking test only; not a trained CAD policy and not a 3090 memory test.",
    }
    model.save_pretrained(out / "tiny-model")
    processor.save_pretrained(out / "tiny-model")
    (out / "smoke.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(vision_smoke(args.dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
