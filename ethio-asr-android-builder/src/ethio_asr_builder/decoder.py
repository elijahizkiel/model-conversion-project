import numpy as np

class CTCDecoder:
    def __init__(self, tokenizer, blank_id=None):
        self.tokenizer = tokenizer
        self.blank_id = blank_id if blank_id is not None else getattr(tokenizer, "pad_token_id", 408)

    def decode_logits(self, logits):
        """
        logits: numpy array of shape [batch_size, time_frames, vocab_size]
        or [time_frames, vocab_size]
        """
        if logits.ndim == 2:
            logits = np.expand_dims(logits, axis=0)

        batch_results = []
        for seq in logits:
            pred_ids = np.argmax(seq, axis=-1)
            # Collapse repeated consecutive tokens
            collapsed = []
            prev_token = None
            for token in pred_ids:
                if token != prev_token:
                    collapsed.append(token)
                    prev_token = token

            # Remove CTC blank tokens
            final_tokens = [tok for tok in collapsed if tok != self.blank_id]

            # Decode to Ethiopic Unicode text via tokenizer
            text = self.tokenizer.decode(final_tokens)
            batch_results.append(text)

        return batch_results
