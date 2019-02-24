import torch
import torch.nn.functional as F
from unittest import TestCase
from torch_transformer import EncoderDecoder


class TestTransformer(TestCase):

    def test_init(self):
        net = EncoderDecoder(
            in_features=16,
            hidden_features=64,
            encoder_num=3,
            decoder_num=4,
            head_num=4,
            attention_activation=None,
            feed_forward_activation=F.relu,
            dropout_rate=0.1,
        )
        print(net)
        encoder_input = torch.randn(3, 5, 16)
        decoder_input = torch.randn(3, 7, 16)
        result = net(encoder_input, None, decoder_input)
