vehicle-maintenance-ai/
│
├── data/
│   ├── raw/
│   │   └── fleet_data.csv
│   └── processed/
│       └── cleaned_data.csv
│
├── ml/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── agent/
│   ├── state.py          # Agent state schema
│   ├── workflow.py       # LangGraph graph
│   ├── planner.py        # Decision logic
│   └── prompts.py
│
├── rag/
│   ├── documents/
│   │   └── manuals.pdf
│   ├── embeddings.py
│   ├── retriever.py
│   └── vector_store.py
│
├── ui/
│   └── app.py             # Streamlit / Gradio
│
├── utils/
│   ├── config.py
│   └── helpers.py
│
├── requirements.txt
├── README.md
└── architecture.png
