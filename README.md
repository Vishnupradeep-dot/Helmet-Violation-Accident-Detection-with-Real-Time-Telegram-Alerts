
🛡️ Helmet Violation & Accident Detection System

An AI-powered real-time safety monitoring platform with YOLOv8, Agent-based RAG LLM, Cloud Logging, and Telegram Alerts.

📌 Project Overview

This project integrates Helmet Violation Detection and Accident Detection into a single intelligent platform. Using YOLOv8 for object detection and Streamlit for visualization, it enables real-time monitoring and proactive alerts for enhanced road safety.

Key highlights:

🚦 Helmet Violation Detection – Detects riders without helmets in real time

🚗 Accident Detection – Identifies potential road accidents instantly

🤖 Agent-based RAG LLM – Provides contextual insights using a Retrieval-Augmented AI Agent

☁️ Cloud Logging – Logs events securely for later analysis

📩 Telegram Alerts – Sends instant notifications for violations and accidents

📊 Streamlit Dashboard – Interactive real-time monitoring dashboard

🧠 Tech Stack

Frontend: Streamlit

Backend: Python 3.9, FastAPI (optional for API integration)

AI Models: YOLOv8 (Helmet & Accident Detection)

RAG LLM: Agent-based Retrieval-Augmented Generation

Cloud Logging: Google Cloud / AWS S3

Notifications: Telegram Bot API

Visualization: Matplotlib, OpenCV

🚀 Features

✅ Real-time accident and helmet violation detection
✅ Multi-camera video stream support
✅ Interactive Streamlit dashboard
✅ Automatic violation logging and reporting
✅ Telegram alerts with detected images
✅ Agent-based AI assistance using RAG LLM
✅ Easy integration with CCTV systems

📦 saferide_yolo
 ┣ 📂 datasets            # Training data
 ┣ 📂 runs               # YOLO training & prediction results
 ┣ 📂 weights            # Pre-trained YOLO models
 ┣ 📂 scripts            # Core detection and Streamlit app scripts
 ┣ 📜 requirements.txt   # Required dependencies
 ┣ 📜 main.py            # Streamlit dashboard entry point
 ┣ 📜 telegram_bot.py    # Real-time Telegram notifications
 ┣ 📜 cloud_logger.py    # Cloud logging integration
 ┣ 📜 rag_agent.py       # Agent-based RAG LLM integration
 ┗ 📜 README.md          # Project documentation

📸 Sample Output

Helmet violation detection

Accident detection alert

Streamlit dashboard screenshot

📢 Future Enhancements

🔹 Integration with traffic police database
🔹 Live GPS-based alert system
🔹 Cloud-based analytics dashboard

🤝 Contributing

Pull requests are welcome! Feel free to fork this repository and improve detection accuracy or add new features.
