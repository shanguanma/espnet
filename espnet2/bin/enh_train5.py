#!/usr/bin/env python3
from espnet2.tasks.enh5 import EnhancementTask1


def get_parser():
    parser = EnhancementTask1.get_parser()
    return parser


def main(cmd=None):
    r"""Enhancemnet frontend training.

    Example:

        % python enh_train.py asr --print_config --optim adadelta \
                > conf/train_enh.yaml
        % python enh_train.py --config conf/train_enh.yaml
    """
    EnhancementTask1.main(cmd=cmd)


if __name__ == "__main__":
    main()
