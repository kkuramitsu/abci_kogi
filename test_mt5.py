import logging
import os
import torch

from transformers import (
    MT5ForConditionalGeneration, T5ForConditionalGeneration,
    AutoConfig, AutoModel, AutoTokenizer,
)

from train_common import parse_hparams, load_DataSet

# GPU利用有無
USE_GPU = torch.cuda.is_available()


def make_generate(model, tokenizer):
    def greedy_search(s: str, max_length=128) -> str:
        input_ids = tokenizer.encode_plus(
            s,
            add_special_tokens=True,
            max_length=max_length,
            padding="do_not_pad",
            truncation=True,
            return_tensors='pt').input_ids.to(model.device)
        greedy_output = model.generate(input_ids, max_length=max_length)
        return tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    return greedy_search


def _main():
    init_dict = dict(
        output_dir='./model',  # path to save the checkpoints
        model_name_or_path='',
        tokenizer_name_or_path='',
        #additional_tokens='<nl> <tab> <b> </b> <e0> <e1> <e2> <e3>',
        seed=42,
        encoding='utf_8',
        column=0, target_column=1,
        kfold=5,  # cross validation
        max_length=128,
        target_max_length=128,
        # da
        da_choice=0.4, da_shuffle=0.3, bos_token='',
        # unsupervised training option
        masking=False,
        masking_ratio=0.35,
        masking_style='denoising',
        # training
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        batch_size=8,
        num_workers=4,  # os.cpu_count(),
        # train_batch_size=8,
        save_checkpoint=False,
        progress_bar=False,
        # eval_batch_size=8,
        max_epochs=50,
        limit_batches=-1,
        gradient_accumulation_steps=1,  # 16
        n_gpu=1 if USE_GPU else 0,
        early_stop_callback=True,
        # if you want to enable 16-bit training then install apex and set this to true
        fp_16=False,
        opt_level='O2',
        max_grad_norm=1.0,
    )
    hparams = parse_hparams(init_dict, Tokenizer=AutoTokenizer)
    print(hparams)

    # 事前学習済みモデルの読み込み
    tokenizer = hparams.tokenizer
    print(tokenizer)
    # config = AutoConfig.from_pretrained(hparams.model_name_or_path)
    # # config.vocab_size = max(config.vocab_size,
    # #                         tokenizer.vocab_size,
    # #                         hparams.vocab_size)
    # if 'mt5' in hparams.model_name_or_path:
    #     model = MT5ForConditionalGeneration(config)
    # else:
    #     model = T5ForConditionalGeneration(config)
    if 'mt5' in hparams.model_name_or_path:
        model = MT5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    else:
        model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    print(hparams.model_name_or_path)
    print(model)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #model.to(DEVICE)

    print('testing ... ', model.device)
    generate = make_generate(model, tokenizer)

    # 最終エポックのモデルを保存

    for file in hparams.files:
        if file.endswith('.tsv'):
            test_data = load_DataSet(hparams, [file])
            if '/' in file:
                _, _, file = file.rpartition('/')
            test_data.test_and_save(
                generate, f'result_{hparams.project}_{file}', max=1000)


if __name__ == '__main__':
    _main()
