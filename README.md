# README.md

## Kyle Watson Financial Tracker

### **Project Overview**
An interactive finance tracker built with Streamlit that processes bank statements (CSV/PDF), categorises transactions, visualises summaries, and sends email alerts.

### **Features**
- Upload CSV or PDF statements
- Data cleaning and categorisation
- Pie chart visualisation of expenses
- Download processed data as Excel
- Email alerts (Gmail SMTP)

---

### ⚙**Setup Instructions**

1. **Clone the repository:**
```bash
git clone <your-repo-link>
cd <repo-folder>
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run locally:**
```bash
streamlit run Fin.py
```

---

### ☁**Deploy to Streamlit Cloud**

1. Push all files (`Fin.py`, `requirements.txt`) to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Connect your GitHub repository.
4. Set the **main file** as `Fin.py`.
5. **Add Secrets** under Settings ➔ Secrets:
```
email_user: kylekkwa@gmail.com
email_pass: your_app_password
smtp_server: smtp.gmail.com
smtp_port: 587
```

6. Deploy your app and share your custom link.

---

### **Important Notes:**
- For Gmail, create an **App Password** instead of using your main password for security.
- Ensure pdfkit dependencies are installed on the server if you expand PDF export functionality.

---

### **Author**
**Kyle Watson Financial Tracker**

---

Your **README.md is ready**. Let me know if you want a logo integration or badge design before you publish to GitHub today.

---

**End of README.md**

---

If you would like, I can package all files as a downloadable zip for your immediate deployment today.
