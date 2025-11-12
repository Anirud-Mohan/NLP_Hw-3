import torch
import torch.nn as nn


class SentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        arch_type='lstm',
        activation='relu',
        bidirectional=False
    ):
        """
        Flexible RNN-based sentiment classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of hidden layers
            num_layers: Number of RNN layers
            dropout: Dropout probability
            arch_type: 'rnn', 'lstm', or 'gru'
            activation: 'relu', 'tanh', or 'sigmoid'
            bidirectional: Whether to use bidirectional RNN
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.arch_type = arch_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.activation = get_activation(activation)

        if self.arch_type == 'lstm':
            self._init_lstm_weights()
        elif self.arch_type == 'rnn':
            self._init_rnn_weights()
        else:
            raise ValueError("Unsupported arch_type. Choose from 'rnn' or 'lstm'.")

        self.dropout = nn.Dropout(dropout)

        output_size = self.hidden_size * self.num_directions
        self.fc_out = nn.Linear(output_size, 1)
        
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x)  
        x = x.transpose(0, 1)  

        # Initial hidden states
        h_forward = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        if self.bidirectional:
            h_backward = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
    
        if self.arch_type == 'lstm':
            c_forward = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            if self.bidirectional:
                c_backward = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        # Process forward direction
        for t in range(seq_len):
            input_t = x[t]
            for layer in range(self.num_layers):
                if layer == 0:
                    layer_input = input_t
                else:
                    if self.bidirectional:
                        layer_input = torch.cat([h_forward[layer - 1], h_backward[layer - 1]], dim=1)
                    else:
                        layer_input = h_forward[layer - 1]

                if self.arch_type == 'rnn':
                    W_ih = self.rnn_layers[layer]['W_ih']
                    W_hh = self.rnn_layers[layer]['W_hh']
                    h_forward[layer] = self.activation(W_ih(layer_input) + W_hh(h_forward[layer]))
                    if layer < self.num_layers - 1:
                        h_forward[layer] = self.dropout(h_forward[layer])
                    
                elif self.arch_type == 'lstm':
                    W_i = self.lstm_layers[layer]['W_i']
                    W_f = self.lstm_layers[layer]['W_f']
                    W_o = self.lstm_layers[layer]['W_o']
                    W_c = self.lstm_layers[layer]['W_c']
                    combined = torch.cat([layer_input, h_forward[layer]], dim=1)
                    i_t = torch.sigmoid(W_i(combined))
                    f_t = torch.sigmoid(W_f(combined))
                    o_t = torch.sigmoid(W_o(combined))
                    g_t = self.activation(W_c(combined))
                    c_forward[layer] = f_t * c_forward[layer] + i_t * g_t
                    h_forward[layer] = o_t * self.activation(c_forward[layer])
                    if layer < self.num_layers - 1:
                        h_forward[layer] = self.dropout(h_forward[layer])

        # Process backward direction (if bidirectional)
        if self.bidirectional:
            for t in range(seq_len - 1, -1, -1):
                input_t = x[t]
                for layer in range(self.num_layers):
                    if layer == 0:
                        layer_input = input_t
                    else:
                        layer_input = torch.cat([h_forward[layer - 1], h_backward[layer - 1]], dim=1)

                    if self.arch_type == 'rnn':
                        W_ih = self.rnn_layers[layer]['W_ih_reverse']
                        W_hh = self.rnn_layers[layer]['W_hh_reverse']
                        h_backward[layer] = self.activation(W_ih(layer_input) + W_hh(h_backward[layer]))
                        if layer < self.num_layers - 1:
                            h_backward[layer] = self.dropout(h_backward[layer])
                        
                    elif self.arch_type == 'lstm':
                        W_i = self.lstm_layers[layer]['W_i_reverse']
                        W_f = self.lstm_layers[layer]['W_f_reverse']
                        W_o = self.lstm_layers[layer]['W_o_reverse']
                        W_c = self.lstm_layers[layer]['W_c_reverse']
                        combined = torch.cat([layer_input, h_backward[layer]], dim=1)
                        i_t = torch.sigmoid(W_i(combined))
                        f_t = torch.sigmoid(W_f(combined))
                        o_t = torch.sigmoid(W_o(combined))
                        g_t = self.activation(W_c(combined))
                        c_backward[layer] = f_t * c_backward[layer] + i_t * g_t
                        h_backward[layer] = o_t * self.activation(c_backward[layer])
                        if layer < self.num_layers - 1:
                            h_backward[layer] = self.dropout(h_backward[layer])

        # Concatenate forward and backward hidden states
        if self.bidirectional:
            out = torch.cat([h_forward[-1], h_backward[-1]], dim=1)  
        else:
            out = h_forward[-1]  
    
        out = self.fc_out(out) 
        out = torch.sigmoid(out)  
        return out
    
    def _init_lstm_weights(self):
        self.lstm_layers = nn.ModuleList()
        
        for layer in range(self.num_layers):
            input_size = self.embedding_dim if layer == 0 else self.hidden_size * self.num_directions
            
            # Forward direction (4 gates: input, forget, output, cell)
            layer_weights = nn.ModuleDict({
                'W_i': nn.Linear(input_size + self.hidden_size, self.hidden_size),
                'W_f': nn.Linear(input_size + self.hidden_size, self.hidden_size),
                'W_o': nn.Linear(input_size + self.hidden_size, self.hidden_size),
                'W_c': nn.Linear(input_size + self.hidden_size, self.hidden_size),
            })
            
            # Backward direction (if bidirectional)
            if self.bidirectional:
                layer_weights['W_i_reverse'] = nn.Linear(input_size + self.hidden_size, self.hidden_size)
                layer_weights['W_f_reverse'] = nn.Linear(input_size + self.hidden_size, self.hidden_size)
                layer_weights['W_o_reverse'] = nn.Linear(input_size + self.hidden_size, self.hidden_size)
                layer_weights['W_c_reverse'] = nn.Linear(input_size + self.hidden_size, self.hidden_size)
            
            self.lstm_layers.append(layer_weights)

    def _init_rnn_weights(self):
        self.rnn_layers = nn.ModuleList()
        
        for layer in range(self.num_layers):
            input_size = self.embedding_dim if layer == 0 else self.hidden_size * self.num_directions
            
            # Forward direction
            layer_weights = nn.ModuleDict({
                'W_ih': nn.Linear(input_size, self.hidden_size, bias=True),
                'W_hh': nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            })
            
            self.rnn_layers.append(layer_weights)

def get_activation(activation_name):
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid()
    }
    return activations.get(activation_name.lower())

