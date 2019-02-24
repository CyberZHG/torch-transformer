import torch
import torch.nn.functional as F
from unittest import TestCase
from torch_transformer import Transformer


class TestTransformer(TestCase):

    def test_init(self):
        net = Transformer(
            encoder_num_embedding=13,
            decoder_num_embedding=17,
            embedding_dim=16,
            encoder_num=3,
            decoder_num=4,
            head_num=4,
            attention_activation=None,
            feed_forward_activation=F.relu,
            dropout_rate=0.1,
        )
        print(net)
        encoder_input = torch.randint(0, 13, (3, 5)).type(torch.LongTensor)
        decoder_input = torch.randint(0, 17, (3, 7)).type(torch.LongTensor)
        net(encoder_input, decoder_input)
