# AI Chatbot Application

A versatile chatbot application with two implementations: one supporting cloud based AI providers (OpenAI & Google Gemini) and another using local AI models via Ollama.

## Features

### App.py - Cloud AI Chatbot
- **Multi-Provider Support**: Switch between OpenAI and Google Gemini
- **Model Flexibility**: Choose from various models (GPT-4o-mini, Gemini-2.5-flash, etc.)
- **Advanced Settings**: 
  - Temperature control (0.0 - 1.0)
  - Top-p sampling (0.0 - 1.0)
  - Top-k sampling for Gemini (1 - 100)
  - Max tokens configuration (50 - 2048)
- **Budget Tracking**: Real-time cost monitoring with visual indicators
- **Chat History**: Persistent conversation memory
- **Smart Summarization**: Generate concise summaries of conversations

### Chat.py - Local AI Chatbot
- **Local Models**: Run AI models locally using Ollama
- **Model Selection**: Choose between llama3.2 and deepseek-r1:1.5b
- **Memory Management**: 
  - Configurable conversation history length
  - Automatic memory trimming
  - Context size optimization (1024 - 16384)
- **LangChain Integration**: Leverages LangChain for conversation management
- **Privacy First**: All processing happens locally


## Prerequisites

### For app.py (Cloud Chatbot)
- Python 3.8+
- OpenAI API key (optional, for OpenAI models)
- Google Gemini API key (optional, for Gemini models)

### For chat.py (Local Chatbot)
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Downloaded models (llama3.2, deepseek-r1:1.5b)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **For local chatbot (chat.py)**: Install and setup Ollama
   ```bash
   # Download and install Ollama from https://ollama.ai/
   
   # Pull the models
   ollama pull llama3.2
   ollama pull deepseek-r1:1.5b
   ```

## Configuration

### Cloud Chatbot (app.py)

1. **Create a `.env` file** in the chatbot directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

2. **API Keys**:
   - Get OpenAI API key from: https://platform.openai.com/api-keys
   - Get Gemini API key from: https://makersuite.google.com/app/apikey

### Local Chatbot (chat.py)

No additional configuration needed! Just ensure Ollama is running:
```bash
ollama serve
```

## Usage

### Running the Cloud Chatbot (app.py)

```bash
streamlit run app.py
```

### Running the Local Chatbot (chat.py)

```bash
streamlit run chat.py
```

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [Streamlit](https://streamlit.io/) - For the amazing web framework
- [OpenAI](https://openai.com/) - For GPT models
- [Google](https://ai.google.dev/) - For Gemini models
- [Ollama](https://ollama.ai/) - For local model inference
- [LangChain](https://langchain.com/) - For conversation management tools
