import os
from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
from transformers import EncoderDecoderModel
import argparse

cache_dir = './cache'
model_name = 'nguyenvulebinh/envibert'

def download_tokenizer_files():
    resources = ['envibert_tokenizer.py', 'dict.txt', 'sentencepiece.bpe.model']
    for item in resources:
        if not os.path.exists(os.path.join(cache_dir, item)):
            tmp_file = hf_bucket_url(model_name, filename=item)
            tmp_file = cached_path(tmp_file, cache_dir=cache_dir)
            os.rename(tmp_file, os.path.join(cache_dir, item))

def init_tokenizer():
    download_tokenizer_files()
    tokenizer = SourceFileLoader("envibert.tokenizer",
                                 os.path.join(cache_dir,
                                              'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)
    return tokenizer

def init_model():
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    download_tokenizer_files()
    tokenizer = SourceFileLoader("envibert.tokenizer",
                                 os.path.join(cache_dir,
                                              'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)
    # set encoder decoder tying to True
    roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name,
                                                                         model_name,
                                                                         tie_encoder_decoder=False)

    # set special tokens
    roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id
    roberta_shared.config.eos_token_id = tokenizer.eos_token_id
    roberta_shared.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    # set decoding params
    roberta_shared.config.max_length = 100
    roberta_shared.config.early_stopping = True
    roberta_shared.config.no_repeat_ngram_size = 3
    roberta_shared.config.length_penalty = 2.0
    roberta_shared.config.num_beams = 1
    roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

    return roberta_shared, tokenizer

trained_model, tokenizer = init_model()
trained_model = trained_model.from_pretrained("datnth1709/VietAI-NLP-ITN")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='The text to translate')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_opt()
    SRC_MAX_LENGTH = 100
    TGT_MAX_LENGTH = 100

    with open(args.file, mode='r', encoding='utf-8') as f:
        text = f.read()

    print("Text input: ")
    print(text)
    print(100*"-")

    input_ids = tokenizer(text, max_length=SRC_MAX_LENGTH, truncation=True, return_tensors='pt')["input_ids"]
    # generate text without beam-search
    outputs = trained_model.generate(
        input_ids, 
        max_length=SRC_MAX_LENGTH, 
        num_return_sequences=1, 
        early_stopping=True
    )

    print("Text predict: ")
    for i, output in enumerate(outputs):
        output_pieces = tokenizer.convert_ids_to_tokens(output.numpy().tolist())
        output_text = tokenizer.sp_model.decode(output_pieces)
        print(output_text)