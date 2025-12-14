---
title: Medical Triage Rag
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.52.1
app_file: streamlit_app.py
pinned: false
---

# Medical Triage RAG System

An AI-powered medical triage and question-answering system that combines symptom assessment with Retrieval-Augmented Generation (RAG) to provide accurate medical information and emergency detection.

## 🚀 What is This Project?

This application uses advanced AI to:

- **Triage patient symptoms** and assess urgency levels
- **Answer medical questions** using a knowledge base of 47,441+ medical Q&A pairs
- **Detect emergencies** and provide appropriate recommendations
- **Retrieve relevant medical information** using semantic search with ChromaDB

The system leverages the MedQuAD dataset and uses the Medical-Llama3-8B model via Hugging Face's Inference API to generate contextually accurate medical responses.

## 🏗️ How It Works

### Architecture

```
User Input → Symptom Classification → Triage Assessment → Emergency Detection
                                                ↓
                                          Response Generation
                                                ↓
Question → Embedding → ChromaDB Search → Context Retrieval → LLM Answer
```

### Key Components

1. **Symptom Triage System**

   - Classifies complaints into categories (chest pain, breathing issues, trauma, etc.)
   - Asks targeted follow-up questions based on symptom category
   - Calculates urgency scores to detect emergencies

2. **RAG (Retrieval-Augmented Generation)**

   - Uses SentenceTransformer (`all-MiniLM-L6-v2`) for embedding medical questions
   - Stores 47,441+ medical Q&A pairs in ChromaDB vector database
   - Retrieves top-k most relevant passages for context
   - Generates answers using Medical-Llama3-8B model

3. **UI Features**
   - Multi-tab interface (Triage Intake, Ask a Question, Dataset Explorer)
   - Dark theme with gradient background
   - Real-time triage assessment
   - Expandable sections for citations and retrieved passages

## 📋 Prerequisites

- Python 3.11+
- Git with Git LFS (for large files)
- Hugging Face account and API token

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shanmukh1315/medical-triage-rag.git
cd medical-triage-rag
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_token_here
HF_MODEL_ID=ruslanmv/Medical-Llama3-8B
```

To get your Hugging Face token:

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "read" permissions
3. Copy and paste it into your `.env` file

### 5. Build the ChromaDB Index (Optional)

The repository includes a pre-built ChromaDB index. To rebuild it from scratch:

```bash
python scripts/build_index.py
```

This will:

- Download the MedQuAD dataset
- Generate embeddings for all Q&A pairs
- Store them in `assets/chroma/`

## 🚀 Running the Application

### Start the Streamlit Server

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Alternative: Using the Hosted Version

Visit the live deployment: [Medical Triage RAG on Hugging Face Spaces](https://huggingface.co/spaces/Shanmukha1315/medical-triage-rag)

## 📱 How to Use the Website

### Tab 1: Triage Intake

1. **Enter Your Complaint**

   - Type your main symptom or health concern in the text box
   - Click "Submit Complaint"

2. **Answer Follow-up Questions**

   - The system will ask 3-5 targeted questions based on your complaint
   - Select answers from the radio button options
   - Submit each answer to proceed

3. **View Your Triage Result**
   - **🚨 Emergency**: Seek immediate medical attention (call 911)
   - **⚠️ Urgent**: Visit urgent care within 24 hours
   - **📅 Non-Urgent**: Schedule regular doctor appointment
   - **🏠 Self-Care**: Rest and over-the-counter remedies may help

### Tab 2: Ask a Question

1. **Type Your Medical Question**

   - Enter any health-related question in the text area
   - Be specific for better results

2. **Get AI-Generated Answer**

   - The system retrieves relevant passages from the medical database
   - Medical-Llama3-8B generates a comprehensive answer
   - View citations and source passages in expandable sections

3. **Explore Retrieved Context**
   - Click "Citations & Sources" to see where the information came from
   - Click "Retrieved Passages" to view the top-4 most relevant medical texts

### Tab 3: Dataset Explorer

- Browse the 47,441+ medical Q&A pairs in the database
- Understand the scope and coverage of the knowledge base
- View sample questions and answers from different medical categories

## 🛠️ Project Structure

```
medical-triage-rag/
├── streamlit_app.py          # Main Streamlit application
├── app.py                    # Alternative Flask/FastAPI app (if needed)
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── assets/
│   └── chroma/              # ChromaDB vector database
│       ├── chroma.sqlite3   # SQLite database (Git LFS)
│       └── 5203349c-.../    # Vector index files (Git LFS)
├── scripts/
│   ├── build_index.py       # Script to build ChromaDB index
│   └── rag_eval.py          # RAG evaluation metrics
└── outputs/
    ├── exp1_relevance_k4.*  # Evaluation results (k=4)
    └── exp2_topk_comparison.* # Top-k comparison analysis
```

## 🧪 Evaluation & Testing

Run the RAG evaluation script to test retrieval quality:

```bash
python scripts/rag_eval.py
```

This generates:

- Relevance metrics (precision, recall, F1)
- Top-k comparison results
- Performance analysis

## ⚙️ Configuration

### Environment Variables

- `HF_TOKEN`: Your Hugging Face API token
- `HF_MODEL_ID`: The model to use (default: `ruslanmv/Medical-Llama3-8B`)

### Customization

Edit `streamlit_app.py` to:

- Adjust triage questions and scoring
- Modify UI styling (CSS in the file)
- Change retrieval parameters (top-k, distance threshold)
- Update emergency detection thresholds

## 🚨 Important Disclaimers

⚠️ **This is not a substitute for professional medical advice**

- Always consult a healthcare provider for medical concerns
- In emergencies, call 911 or go to the nearest emergency room
- This tool is for informational and educational purposes only
- AI-generated responses may contain inaccuracies

## 📊 Dataset

This project uses the **MedQuAD** (Medical Question Answering Dataset):

- 47,441+ medical Q&A pairs
- Sourced from trusted health organizations
- Covers multiple medical specialties

## 🤖 Models Used

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

  - Fast, accurate semantic embeddings
  - 384-dimensional vectors

- **Language Model**: `ruslanmv/Medical-Llama3-8B`
  - Specialized medical question answering
  - Fine-tuned on medical literature

## 🔒 Privacy & Security

- No patient data is stored permanently
- All processing happens in-memory during the session
- Hugging Face API calls are secured with your token
- No identifiable health information is logged

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with medical data regulations (HIPAA, GDPR) if deploying in production.

## 🙏 Acknowledgments

- MedQuAD dataset creators
- Hugging Face for model hosting
- Streamlit for the web framework
- ChromaDB for vector storage

## 📞 Support

For issues or questions:

- Open an issue on [GitHub](https://github.com/shanmukh1315/medical-triage-rag/issues)
- Check existing documentation and discussions

---

**Built with ❤️ for better healthcare accessibility**
