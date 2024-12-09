import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from model import *
from dotenv import load_dotenv
import os

load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model('models/model_layers2_dropout05_augData_NoName.pth', device)

async def start(update: Update, context):
    """Send a welcome message when the /start command is issued."""
    welcome_text = (
        f"Hello, {update.effective_user.first_name}! Welcome to HeinleinAI, "
        "your ultimate guide to Luna-City, humanity's first settlement on the Moon!\n\n"
    )
    await update.message.reply_text(welcome_text)
    await help_command(update, context)

async def help_command(update: Update, context):
    """Send a help message when the /help command is issued."""
    help_text = (
        "Here are all the commands available!\n\n"
        "/start - Start the bot and get a welcome message\n"
        "/help - Get a list of available commands and their descriptions\n"
        "\nFeel free to ask anything!"
    )
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context):
    """Handle user messages."""
    user_message = update.message.text
    print(f"Received message: {user_message}")

    # Use the model to generate an answer
    answer = translate(model, user_message, train_question_vocab, train_answer_vocab, device=device)
    
    # Send the model's response back to the user
    await update.message.reply_text(answer)

def main():
    """Start the bot."""
    app = ApplicationBuilder().token(API_TOKEN).build()

    # Command Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    
    # Message Handler for responding to user queries
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == '__main__':
    main()
