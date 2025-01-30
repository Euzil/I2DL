import torch
from .tokenizer import load_pretrained_fast


class CustomCollator:

    def __init__(self, tokenizer=None, max_length=None):
        """

        Args:
            tokenizer: tokenizer used to create the encodings
            max_length: If set, truncates sequences to have a maximum length

        Attributes:
            self.truncation: Truncation mode
        """
        self.max_length = max_length
        self.truncation = 'do_not_truncate' if max_length is None else 'longest_first'
        if tokenizer is None:
            tokenizer = load_pretrained_fast()
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """
        Transforms list of input sentences into tensors containing the token ids. The output is a dictionary containing
        the encoder and decoder inputs as well as the prepared masks for pad masking. Apart from that, the output dict
        also contains the corresponding labels, as well as their mask and lengths.

        Args:
            batch: list of size batch_size containing dictionaries with keys 'source' and 'target'

        """
        source_encodings = self.tokenizer.batch_encode_plus([b['source'] for b in batch],
                                                            padding=True,
                                                            max_length=self.max_length,
                                                            truncation=self.truncation)

        target_encodings = self.tokenizer.batch_encode_plus([b['target'] for b in batch],
                                                            padding=True,
                                                            max_length=self.max_length,
                                                            truncation=self.truncation)

        return {
            'encoder_inputs': torch.tensor(source_encodings['input_ids']),
            'encoder_mask': torch.tensor(source_encodings['attention_mask']).unsqueeze_(-2).bool(),
            'decoder_inputs': torch.tensor(target_encodings['input_ids'])[:, :-1],
            'decoder_mask': torch.tensor(target_encodings['attention_mask'])[:, 1:].unsqueeze_(-2).bool(),
            'labels': torch.tensor(target_encodings['input_ids'])[:, 1:],
            'label_mask': torch.tensor(target_encodings['attention_mask'])[:, 1:].bool(),
            'label_length': torch.tensor(target_encodings['attention_mask'])[:, 1:].sum(dim=-1)}
