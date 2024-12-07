import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import random
from sklearn.model_selection import train_test_split

# Load data
with open('data/paraphrased_dataset.json', 'r') as f:
    data = json.load(f)

questions = [entry["question"] for entry in data["questions"]]
answers = [entry["answer"] for entry in data["questions"]]

# Tokenizer
def tokenize(text):
    return text.lower().split()

# Build Vocabulary
def build_vocab(sentences):
    vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    for sentence in sentences:
        for token in tokenize(sentence):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

# Train/Validation Split
train_questions, val_questions, train_answers, val_answers = train_test_split(questions, answers, test_size=0.2, random_state=42)

# Build vocabularies from training data
train_question_vocab = build_vocab(train_questions)
train_answer_vocab = build_vocab(train_answers)

PAD_IDX = train_question_vocab["<pad>"]
BOS_IDX = train_question_vocab["<bos>"]
EOS_IDX = train_question_vocab["<eos>"]

# Encode Sentence
def encode_sentence(sentence, vocab):
    return [vocab["<bos>"]] + [vocab.get(token, vocab["<unk>"]) for token in tokenize(sentence)] + [vocab["<eos>"]]

# Decode Sentence
def decode_sentence(tokens, vocab):
    reverse_vocab = {idx: token for token, idx in vocab.items()}
    return " ".join(reverse_vocab[token] for token in tokens if token not in {PAD_IDX, BOS_IDX, EOS_IDX})

# Dataset
class QADataset(Dataset):
    def __init__(self, questions, answers, q_vocab, a_vocab):
        self.questions = [encode_sentence(q, q_vocab) for q in questions]
        self.answers = [encode_sentence(a, a_vocab) for a in answers]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return torch.tensor(self.questions[idx]), torch.tensor(self.answers[idx])

def collate_fn(batch):
    questions, answers = zip(*batch)
    questions = nn.utils.rnn.pad_sequence(questions, padding_value=PAD_IDX, batch_first=True)
    answers = nn.utils.rnn.pad_sequence(answers, padding_value=PAD_IDX, batch_first=True)
    return questions, answers

# Update Dataset with the train/validation split
train_dataset = QADataset(train_questions, train_answers, train_question_vocab, train_answer_vocab)
val_dataset = QADataset(val_questions, val_answers, train_question_vocab, train_answer_vocab)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Seq2Seq Model with Dropout
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size, hidden_size, num_layers, pad_idx, dropout=0.5):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.Embedding(input_dim, embed_size, padding_idx=pad_idx)
        self.decoder = nn.Embedding(output_dim, embed_size, padding_idx=pad_idx)

        self.encoder_rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder_rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, src, tgt):
        # Encoder
        embedded_src = self.encoder(src)
        _, (hidden, cell) = self.encoder_rnn(embedded_src)

        # Decoder
        embedded_tgt = self.decoder(tgt)
        outputs, _ = self.decoder_rnn(embedded_tgt, (hidden, cell))

        # Fully connected output
        predictions = self.fc_out(outputs)
        return predictions

# Training Loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)

        # Reshape for loss calculation
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# Validation Loop
def evaluate(model, val_dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in val_dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

    return epoch_loss / len(val_dataloader)

# Inference
def capitalize_first_letter(text):
    # Ensure the first letter is capitalized, while the rest stays as is
    return text[0].upper() + text[1:] if text else text

def translate(model, question, q_vocab, a_vocab, max_len=50, device='cpu'):
    model.eval()

    # Check if the question is empty
    if not question.strip():
        return "I'm sorry, I didn't understand that."

    # Tokenize the question
    tokens = encode_sentence(question, q_vocab)

    # Check if the question contains only special tokens (<bos>, <pad>, <eos>)
    unknown_tokens_count = sum(1 for token in tokens if token == q_vocab["<unk>"])

    # Check if there are 2 or more non-<unk> tokens
    if len(tokens) >= 3 and unknown_tokens_count >= int(len(tokens)/3):
        return "I'm sorry, I didn't understand that."
    elif len(tokens) <= 3 and unknown_tokens_count >= 1:
        return "I'm sorry, I didn't understand that."

    # Encode the question
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        embedded_src = model.encoder(tokens_tensor)
        _, (hidden, cell) = model.encoder_rnn(embedded_src)

    # Start decoding
    tgt_token = BOS_IDX
    output_tokens = []

    for _ in range(max_len):
        tgt = torch.tensor([[tgt_token]]).to(device)
        embedded_tgt = model.decoder(tgt)
        output, (hidden, cell) = model.decoder_rnn(embedded_tgt, (hidden, cell))
        predictions = model.fc_out(output.squeeze(1))
        tgt_token = predictions.argmax(1).item()

        if tgt_token == EOS_IDX:
            break

        output_tokens.append(tgt_token)

    # Decode the generated tokens into a sentence
    answer = decode_sentence(output_tokens, a_vocab)
    
    # Capitalize the first letter of the answer
    return capitalize_first_letter(answer)

# Training and Evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2Seq(
    input_dim=len(train_question_vocab),
    output_dim=len(train_answer_vocab),
    embed_size=256,
    hidden_size=512,
    num_layers=2,
    pad_idx=PAD_IDX
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# for epoch in range(100):
#     train_loss = train(model, train_dataloader, optimizer, criterion, device)
#     val_loss = evaluate(model, val_dataloader, criterion, device)
#     print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# # Save model
# os.makedirs('models', exist_ok=True)
# torch.save(model.state_dict(), 'models/model_layers2_dropout05_augData_NoName.pth')

def load_model(model_path, device='cpu'):
    model = Seq2Seq(
        input_dim=len(train_question_vocab),
        output_dim=len(train_answer_vocab),
        embed_size=256,
        hidden_size=512,
        num_layers=2,
        pad_idx=PAD_IDX
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# # Testing
# model = load_model('models/model_layers2_dropout05_augData.pth', device)
# while True:
#     question = input("Ask a question (or type 'exit' to quit): ")
#     if question.lower() == "exit":
#         break
#     answer = translate(model, question, train_question_vocab, train_answer_vocab, device=device)
#     print(f"Answer: {answer}")
