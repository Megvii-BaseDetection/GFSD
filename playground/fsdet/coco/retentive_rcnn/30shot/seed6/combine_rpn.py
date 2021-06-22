import argparse

import torch


def load_model(model_path):
    try:
        ckpt = torch.load(model_path)
    except RuntimeError:
        ckpt = torch.load(model_path, map_location="cpu")

    if "model" in ckpt.keys():
        return ckpt["model"]
    return ckpt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--novel-model", type=str,
                        required=True,
                        help="path to the final trained model")
    parser.add_argument("--base-model", type=str,
                        default="/path/to/your/model.pth",
                        help="path to the base model")
    parser.add_argument("--save-model", type=str,
                        default="model_redetect.pth",
                        help="path to the saved model")
    return parser.parse_args()


def combine_and_save(
    base_model="/path/to/your/model.pth",
    novel_model="log/model_final.pth",
    save_model="log/model_redetect.pth"
):
    base_model = load_model(base_model)
    novel_model = load_model(novel_model)
    # ensemble rpn
    rpn_cls_keys = [
        'proposal_generator.rpn_head.objectness_logits.weight',
        'proposal_generator.rpn_head.objectness_logits.bias'
    ]
    for k in rpn_cls_keys:
        novel_model[k.replace(".objectness", ".finetuned_objectness")] = \
            novel_model[k]
        novel_model[k] = base_model[k]
    rpn_keys = [k for k in base_model.keys() if "rpn" in k]
    for k in rpn_keys:
        assert torch.all(base_model[k] == novel_model[k]), f"{k} not equal!"

    torch.save(novel_model, save_model)
    return save_model


if __name__ == "__main__":
    args = parse_args()
    combine_and_save(args.base_model, args.novel_model, args.save_model)
