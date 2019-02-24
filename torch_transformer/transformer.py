import torch.nn as nn
import torch.nn.functional as F
from torch_position_embedding import TrigonometricPositionEmbedding
from torch_layer_normalization import LayerNormalization
from torch_multi_head_attention import MultiHeadAttention
from torch_embed_sim import EmbeddingSim
from .feed_forward import FeedForward


__all__ = [
    'BlockWrapper',
    'EncoderComponent', 'DecoderComponent',
    'Encoder', 'Decoder',
    'EncoderDecoder', 'Transformer',
]


class BlockWrapper(nn.Module):

    def __init__(self,
                 in_features,
                 layer,
                 dropout_rate=0.0):
        """Wrap layer with add and normalization.

        :param in_features: Length of input features.
        :param layer: The layer to be wrapped.
        :param dropout_rate: Dropout rate.
        """
        super(BlockWrapper, self).__init__()
        self.in_features = in_features
        self.layer = layer
        self.dropout_layer = None
        if dropout_rate > 0:
            self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.normal_layer = LayerNormalization(normal_shape=in_features)

    def forward(self, x, *args, **kwargs):
        y = self.layer(x, *args, **kwargs)
        if self.dropout_layer is not None:
            y = self.dropout_layer(y)
        return self.normal_layer(x + y)


class EncoderComponent(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        """Encoder component.

        :param in_features: Length of the input features.
        :param hidden_features: Number of features inside feed-forward layer.
        :param head_num: Number of heads.
        :param attention_activation: Activation for attention layer.
        :param feed_forward_activation: Activation for feed-forward layer.
        :param dropout_rate: Dropout rate.
        """
        super(EncoderComponent, self).__init__()
        self.attention = BlockWrapper(
            in_features,
            layer=MultiHeadAttention(
                in_features=in_features,
                head_num=head_num,
                activation=attention_activation,
            ),
            dropout_rate=dropout_rate,
        )
        self.feed_forward = BlockWrapper(
            in_features,
            layer=FeedForward(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=in_features,
                activation=feed_forward_activation,
            ),
            dropout_rate=dropout_rate,
        )

    def forward(self, x, mask=None):
        return self.feed_forward(self.attention(x, x, x, mask=mask))


class DecoderComponent(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        """Decoder component.

        :param in_features: Length of the input features.
        :param hidden_features: Number of features inside feed-forward layer.
        :param head_num: Number of heads.
        :param attention_activation: Activation for attention layer.
        :param feed_forward_activation: Activation for feed-forward layer.
        :param dropout_rate: Dropout rate.
        """
        super(DecoderComponent, self).__init__()
        self.self_attention = BlockWrapper(
            in_features,
            layer=MultiHeadAttention(
                in_features=in_features,
                head_num=head_num,
                activation=attention_activation,
            ),
            dropout_rate=dropout_rate,
        )
        self.attention = BlockWrapper(
            in_features,
            layer=MultiHeadAttention(
                in_features=in_features,
                head_num=head_num,
                activation=attention_activation,
            ),
            dropout_rate=dropout_rate,
        )
        self.feed_forward = BlockWrapper(
            in_features,
            layer=FeedForward(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=in_features,
                activation=feed_forward_activation,
            ),
            dropout_rate=dropout_rate,
        )

    def forward(self, x, encoded, mask=None):
        y = self.self_attention(x, x, x, mask=MultiHeadAttention.gen_history_mask(x))
        y = self.attention(y, encoded, encoded, mask=mask)
        return self.feed_forward(y)


class Encoder(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 encoder_num,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        """Encoder.

        :param in_features: Length of the input features.
        :param hidden_features: Number of features inside feed-forward layer.
        :param encoder_num: Number of encoder components.
        :param head_num: Number of heads.
        :param attention_activation: Activation for attention layer.
        :param feed_forward_activation: Activation for feed-forward layer.
        :param dropout_rate: Dropout rate.
        """
        super(Encoder, self).__init__()
        self.components = []
        for i in range(encoder_num):
            component = EncoderComponent(
                in_features=in_features,
                hidden_features=hidden_features,
                head_num=head_num,
                attention_activation=attention_activation,
                feed_forward_activation=feed_forward_activation,
                dropout_rate=dropout_rate,
            )
            self.add_module('encoder_%d' % i, component)
            self.components.append(component)

    def forward(self, x, mask=None):
        for component in self.components:
            x = component(x, mask=mask)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 decoder_num,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        """Decoder.

        :param in_features: Length of the input features.
        :param hidden_features: Number of features inside feed-forward layer.
        :param decoder_num: Number of decoder components.
        :param head_num: Number of heads.
        :param attention_activation: Activation for attention layer.
        :param feed_forward_activation: Activation for feed-forward layer.
        :param dropout_rate: Dropout rate.
        """
        super(Decoder, self).__init__()
        self.components = []
        for i in range(decoder_num):
            component = DecoderComponent(
                in_features=in_features,
                hidden_features=hidden_features,
                head_num=head_num,
                attention_activation=attention_activation,
                feed_forward_activation=feed_forward_activation,
                dropout_rate=dropout_rate,
            )
            self.add_module('decoder_%d' % i, component)
            self.components.append(component)

    def forward(self, x, encoded, mask=None):
        for component in self.components:
            x = component(x, encoded=encoded, mask=mask)
        return x


class EncoderDecoder(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 encoder_num,
                 decoder_num,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        """Encoder and decoder.

        :param in_features: Length of the input features.
        :param hidden_features: Number of features inside feed-forward layer.
        :param encoder_num: Number of encoder components.
        :param decoder_num: Number of decoder components.
        :param head_num: Number of heads.
        :param attention_activation: Activation for attention layer.
        :param feed_forward_activation: Activation for feed-forward layer.
        :param dropout_rate: Dropout rate.
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(
            in_features=in_features,
            hidden_features=hidden_features,
            encoder_num=encoder_num,
            head_num=head_num,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
        )
        self.decoder = Decoder(
            in_features=in_features,
            hidden_features=hidden_features,
            decoder_num=decoder_num,
            head_num=head_num,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
        )

    def forward(self, encoder_input, decoder_input, encoder_mask=None):
        encoded = self.encoder(encoder_input, encoder_mask)
        return self.decoder(decoder_input, encoded, mask=encoder_mask)


class Transformer(nn.Module):

    def __init__(self,
                 encoder_num_embedding,
                 decoder_num_embedding,
                 embedding_dim,
                 encoder_num,
                 decoder_num,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu,
                 dropout_rate=0.0):
        """Transformer.

        :param encoder_num_embedding: Number of tokens for encoder input.
        :param decoder_num_embedding: Number of tokens for decoder input.
        :param embedding_dim: The dimension of embeddings.
        :param encoder_num: Number of encoder components.
        :param decoder_num: Number of decoder components.
        :param head_num: Number of heads.
        :param attention_activation: Activation for attention layer.
        :param feed_forward_activation: Activation for feed-forward layer.
        :param dropout_rate: Dropout rate.
        """
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(
            num_embeddings=encoder_num_embedding,
            embedding_dim=embedding_dim,
        )
        self.decoder_embedding = nn.Embedding(
            num_embeddings=decoder_num_embedding,
            embedding_dim=embedding_dim,
        )
        self.position_embedding = TrigonometricPositionEmbedding(
            embedding_dim=embedding_dim,
            mode=TrigonometricPositionEmbedding.MODE_ADD,
        )
        self.encoder_decoder = EncoderDecoder(
            in_features=embedding_dim,
            hidden_features=embedding_dim * 4,
            encoder_num=encoder_num,
            decoder_num=decoder_num,
            head_num=head_num,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
        )
        self.embedding_sim = EmbeddingSim(num_embeddings=decoder_num_embedding)

    def forward(self, encoder_input, decoder_input, encoder_mask=None):
        encoder_input = self.position_embedding(self.encoder_embedding(encoder_input))
        decoder_input = self.position_embedding(self.decoder_embedding(decoder_input))
        decoded = self.encoder_decoder(encoder_input, decoder_input, encoder_mask)
        return self.embedding_sim(decoded, self.decoder_embedding.weight)
