from torch import nn
import torch
from ..network import SCORE_SAVER
from ..network import Embedding
from ..network import Encoder
from ..network import Decoder


class Transformer(nn.Module):

    def __init__(self,
                vocab_size: int,
                eos_token_id: int,
                hparams: dict = None,
                max_length: int = 2048,
                weight_tying: bool = True):
        """

        Args:
            vocab_size: Number of elements in the vocabulary
            eos_token_id: ID of the End-Of-Sequence Token - used in predict()
            weight_tying: Activate Weight Tying between Input Embedding and Output layer (default=True)
            max_length: Maximum sequence length (default=2048)

        Attributes:
            self.d_model: Dimension of Embedding (default=512)
            self.d_k: Dimension of Keys and Queries (default=64)
            self.d_v: Dimension of Values (default=64)
            self.n_heads: Number of Attention Heads (default=8)
            self.d_ff: Dimension of hidden layer (default=2048)
            self.n: Number of Encoder/Decoder Blocks (default=6)
            self.dropout: Dropout probability (default=0.1)
        """
        super().__init__()

        if hparams is None:
            hparams = {}
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.weight_tying = weight_tying

        self.d_model = hparams.get('d_model', 512)
        self.d_k = hparams.get('d_k', 64)
        self.d_v = hparams.get('d_v', self.d_k)
        self.n_heads = hparams.get('n_heads', 8)
        self.d_ff = hparams.get('d_ff', 2048)
        self.n = hparams.get('n', 6)
        self.dropout = hparams.get('dropout', 0.1)

        self.hparams = {
            'd_model': self.d_model,
            'd_k': self.d_k,
            'd_v': self.d_v,
            'd_ff': self.d_ff,
            'n_heads': self.n_heads,
            'n': self.n,
            'dropout': self.dropout
        }

        self.embedding = None
        self.encoder = None
        self.decoder = None
        self.output_layer = None

        ########################################################################
        # TODO:                                                                #
        #   Task 9: Initialize the transformer!                                #
        #            You will need:                                            #
        #               - An embedding layer                                   #
        #               - An encoder                                           #
        #               - A decoder                                            #
        #               - An output layer                                      #
        #                                                                      #
        # Hint 9: Have a look at the output shape of the decoder and the       #
        #          output shape of the transformer model to figure out the     #
        #          dimensions of the output layer! We will not need a bias!    #
        ########################################################################


        # Initialize embedding layer
        self.embedding = Embedding(vocab_size=self.vocab_size, 
                                d_model=self.d_model,
                                max_length=self.max_length)

        # Initialize encoder
        self.encoder = Encoder(d_model=self.d_model,
                            d_k=self.d_k,
                            d_v=self.d_v,
                            n_heads=self.n_heads,
                            d_ff=self.d_ff,
                            n=self.n,
                            dropout=self.dropout)

        # Initialize decoder
        self.decoder = Decoder(d_model=self.d_model,
                            d_k=self.d_k,
                            d_v=self.d_v,
                            n_heads=self.n_heads,
                            d_ff=self.d_ff,
                            n=self.n,
                            dropout=self.dropout)

        # Initialize output layer (linear projection to vocab size)
        # No bias needed as mentioned in the hint
        self.output_layer = nn.Linear(self.d_model, self.vocab_size, bias=False)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        if self.weight_tying:
            self.output_layer.weight = self.embedding.embedding.weight

    def forward(self,
                encoder_inputs: torch.Tensor,
                decoder_inputs: torch.Tensor,
                encoder_mask: torch.Tensor = None,
                decoder_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            encoder_inputs: Encoder Tokens Shape
            decoder_inputs: Decoder Tokens
            encoder_mask: Optional Padding Mask for Encoder Inputs
            decoder_mask: Optional Padding Mask for Decoder Inputs

        Returns:
                torch.Tensor: Logits of the Transformer Model
            
        Shape:
            - encoder_inputs: (batch_size, sequence_length_decoder)
            - decoder_inputs: (batch_size, sequence_length_encoder)
            - encoder_mask: (batch_size, sequence_length_encoder, sequence_length_encoder)
            - decoder_mask: (batch_size, sequence_length_decoder, sequence_length_decoder)
            - outputs: (batch_size, sequence_length_decoder, vocab_size)
        """

        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 9: Implement the forward pass of the transformer!             #
        #            You will need to:                                         #
        #               - Compute the encoder embeddings                       #
        #               - Compute the forward pass through the encoder         #
        #               - Compute the decoder embeddings                       #
        #               - Compute the forward pass through the decoder         #
        #               - Compute the output logits                            #
        #   Task 10: Pass on the encoder and decoder padding masks!            #
        #                                                                      #
        # Hints 10: Have a look at the forward pass of the encoder and decoder #
        #           to figure out which masks to pass on!                      #
        ########################################################################


        # Compute encoder embeddings
        encoder_embedded = self.embedding(encoder_inputs)

        # Pass through encoder with encoder mask
        encoder_output = self.encoder(encoder_embedded, encoder_mask)

        # Compute decoder embeddings
        decoder_embedded = self.embedding(decoder_inputs)

        # Pass through decoder with both masks
        # The decoder needs both the decoder mask (for causal attention)
        # and the encoder mask (for cross attention)
        decoder_output = self.decoder(decoder_embedded, encoder_output, 
                                    decoder_mask, encoder_mask)

        # Compute output logits
        outputs = self.output_layer(decoder_output)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs

    def predict(self,
                    encoder_input: torch.Tensor,
                    max_iteration_length: int = 100,
                    probabilistic: bool = False,
                    return_scores=False) -> tuple:
            """
            Predicts the output sequence given an input sequence using the Transformer model.

            Args:
                encoder_input (torch.Tensor): The input sequence to be encoded.
                max_iteration_length (int, optional): The maximum length of the output sequence. Defaults to 100.
                probabilistic (bool, optional): Whether to sample from the output distribution probabilistically. Defaults to False.
                return_scores (bool, optional): Whether to return the scores recorded during prediction. Defaults to False.

            Shape:
                - encoder_input: (sequence_length, d_model)

            Returns:
                tuple: A tuple containing the predicted output sequence and the recorded scores (if return_scores is True).
            """
            if return_scores:
                SCORE_SAVER.record_scores()

            # The Model only accepts batched inputs, so we have to add a batch dimension
            encoder_input = encoder_input.unsqueeze(0)

            self.eval()
            with torch.no_grad():

                # Compute the encoder embeddings
                encoder_input = self.embedding(encoder_input)

                # Run the embeddings through the encoder
                # We only have to do this once, since the input does not change!
                encoder_output = self.encoder(encoder_input)

                # Initialize the output sequence
                output_sequence = []

                for _ in range(max_iteration_length):

                    # Add the start token (or in our model it is the eos token) to the output sequence
                    # and add a batch dimension
                    decoder_input = torch.tensor([self.eos_token_id] + output_sequence).unsqueeze(0)

                    # Compute the decoder embeddings
                    decoder_input = self.embedding(decoder_input)

                    # Run the embeddings through the decoder
                    output = self.decoder(decoder_input, encoder_output)

                    # Compute the logits of the output layer
                    logits = self.output_layer(output).squeeze(0)

                    # We could run all logits through a softmax and would get the same result
                    # But we are going to just append the last output of the logits
                    # Remember - because of the causal masks, the predictions for the previous outputs never change!
                    last_logit = logits[-1]

                    # If probalistic is True, we sample from the output distribution and append the sample to the output sequence
                    if probabilistic:
                        output_distribution = torch.distributions.Categorical(logits=last_logit)
                        output = output_distribution.sample().item()
                        output_sequence.append(output)

                    # Else we just take the most likely output and append it to the output sequence
                    else:
                        output = torch.argmax(last_logit).item()
                        output_sequence.append(output)

                    # If we predicted the end of sequence token, we stop
                    if output_sequence[-1] is self.eos_token_id:
                        break

            return output_sequence, SCORE_SAVER.get_scores()
