import pdfplumber
import streamlit as st
import pandas as pd
import pdfplumber
import plotly.express as px
import calendar
import smtplib
from email.message import EmailMessage
from io import BytesIO
import tempfile
import os
import datetime

try:
    import pdfkit
    PDFKIT_INSTALLED = True
except ImportError:
    PDFKIT_INSTALLED = False

from fpdf import FPDF

st.set_page_config(page_title="Finance Tracker", layout="wide")


# ---------- Functions ----------

@st.cache_data
def process_csv(file):
    try:
        df = pd.read_csv(file)
        df.rename(columns={
            'TRANS DATE': 'Date',
            'DETAILS': 'Description',
            'TOTAL AMOUNT': 'Amount',
            'TRANS TYPE': 'Category'
        }, inplace=True)
        df = df[['Date', 'Description', 'Amount', 'Category']]
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Category'] = df['Amount'].apply(lambda x: 'Debit' if x < 0 else 'Credit')
        df['Amount'] = df['Amount'].abs()
        return df
    except Exception as e:
        st.error(f"CSV Error: {e}")
        return pd.DataFrame()


@st.cache_data
def process_pdf(file):
    try:
        data = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                for line in text.split('\n'):
                    if "POS" in line or "TRF" in line or "PURCHASE" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            date = parts[0]
                            desc = " ".join(parts[1:-2])
                            amt = parts[-2].replace('J$', '').replace(',', '')
                            cat = 'Debit' if '-' in parts[-2] else 'Credit'
                            data.append([date, desc, abs(float(amt)), cat])
        df = pd.DataFrame(data, columns=['Date', 'Description', 'Amount', 'Category'])
        return df
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return pd.DataFrame()


def classify_expense(description, mapping):
    desc = description.lower()
    for category, keywords in mapping.items():
        for kw in keywords:
            if kw in desc:
                return category
    return 'Uncategorized'


def send_email_alert(receiver_email, subject, body, sender_email, sender_password, smtp_server, smtp_port=587):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False


def export_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Transactions')
        writer.save()
    processed_data = output.getvalue()
    return processed_data


def export_to_pdf(text_report):
    # Try pdfkit first
    if PDFKIT_INSTALLED:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
            f.write(text_report.encode('utf-8'))
            f.flush()
            pdf_file = f.name.replace('.html', '.pdf')
            pdfkit.from_file(f.name, pdf_file)
            with open(pdf_file, 'rb') as pdf_f:
                pdf_bytes = pdf_f.read()
            os.unlink(f.name)
            os.unlink(pdf_file)
            return pdf_bytes
    else:
        # Fallback with fpdf
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text_report.split('\n'):
            pdf.cell(0, 10, line, ln=True)
        return pdf.output(dest='S').encode('latin1')


# ---------- App Start ----------

st.title("📊 Personal Finance Tracker")

uploaded_files = st.file_uploader(
    "Upload CSV or PDF files",
    type=["csv", "pdf"],
    accept_multiple_files=True
)

data = pd.DataFrame()
if uploaded_files:
    for file in uploaded_files:
        fname = file.name.lower()
        if fname.endswith(".csv"):
            data = pd.concat([data, process_csv(file)], ignore_index=True)
        elif fname.endswith(".pdf"):
            data = pd.concat([data, process_pdf(file)], ignore_index=True)

if data.empty:
    st.info("Upload your bank CSV or PDF statements to get started.")
    st.stop()

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])

# -------- Sidebar: User-friendly Categories and Budgets --------

st.sidebar.header("🗂 Customize Categories and Budgets")

# Default categories and keywords
default_mapping = {
    "Food": ["juici", "kfc", "restaurant", "burger", "pizza"],
    "Grocery": ["hi-lo", "supermarket", "wholesale"],
    "Utilities": ["jps", "nwc", "flow", "internet", "light", "water"],
    "Transport": ["uber", "taxi", "gas"],
    "Income": ["remitly", "deposit", "transfer", "payroll"],
    "Miscellaneous": ["atm"],
    "Other": []
}

CATEGORY_KEYWORDS = {}

st.sidebar.markdown("### Edit Categories and Keywords")
for category, keywords in default_mapping.items():
    with st.sidebar.expander(f"{category} Keywords", expanded=False):
        kw_text = st.text_area(
            label=f"Keywords for {category} (comma separated)",
            value=", ".join(keywords),
            key=f"kw_{category}"
        )
        CATEGORY_KEYWORDS[category] = [kw.strip().lower() for kw in kw_text.split(",") if kw.strip()]

st.sidebar.markdown("### 💸 Set Monthly Budgets (J$)")

MONTHLY_BUDGETS = {}

for category in CATEGORY_KEYWORDS.keys():
    default_val = 0
    if category == "Food":
        default_val = 15000
    elif category == "Grocery":
        default_val = 10000
    elif category == "Utilities":
        default_val = 8000
    elif category == "Transport":
        default_val = 6000
    elif category == "Miscellaneous":
        default_val = 5000
    MONTHLY_BUDGETS[category] = st.sidebar.number_input(
        label=f"Budget for {category}",
        min_value=0,
        value=default_val,
        step=500,
        key=f"budget_{category}"
    )

st.sidebar.markdown("### 🎯 Set Monthly Savings Goal (J$)")
default_savings_goal = 5000
savings_goal_input = st.sidebar.number_input(
    "Savings Goal Amount (J$)", min_value=0, value=default_savings_goal, step=500
)
SAVINGS_GOAL = savings_goal_input

# Email notification settings
st.sidebar.header("📧 Email Notification Settings (Optional)")
notify_email = st.sidebar.text_input("Send alerts to email (leave blank to disable):")
if notify_email:
    sender_email = st.sidebar.text_input("Sender Email (SMTP):")
    sender_password = st.sidebar.text_input("Sender Email Password:", type="password")
    smtp_server = st.sidebar.text_input("SMTP Server (e.g. smtp.gmail.com):", value="smtp.gmail.com")
    smtp_port = st.sidebar.number_input("SMTP Port:", min_value=1, max_value=65535, value=587)


# -------------- Income & Spending Trends for ALL DATA (User Friendly) --------------

st.subheader("📉 Income and Spending Trends (All Data)")

def get_current_quarter(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return (dt.month - 1) // 3 + 1

def get_current_half_year(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return 1 if dt.month <= 6 else 2

period_type = st.radio(
    "Select Period Type",
    options=["Monthly (Select 2 months)", "Quarterly (Current Quarter)", "Semi-Annually (Current Half-Year)"]
)

trend = data.copy()
trend['Year'] = trend['Date'].dt.year
trend['Month'] = trend['Date'].dt.month
trend.set_index('Date', inplace=True)

if period_type == "Monthly (Select 2 months)":
    trend['Month-Year'] = trend.index.strftime('%B %Y')
    available_month_years = sorted(trend['Month-Year'].unique(), key=lambda x: datetime.datetime.strptime(x, '%B %Y'))

    selected_months = st.multiselect(
        "Select exactly 2 months to compare",
        options=available_month_years,
        default=available_month_years[-2:] if len(available_month_years) >= 2 else available_month_years
    )

    if len(selected_months) != 2:
        st.warning("Please select exactly 2 months.")
        st.stop()

    filtered_trend = trend[trend['Month-Year'].isin(selected_months)]
    agg = filtered_trend.groupby(['Month-Year', 'Category'])['Amount'].sum().unstack(fill_value=0)
    agg['Net Flow'] = agg.get('Credit', 0) - agg.get('Debit', 0)
    agg = agg.reset_index()

    fig = px.bar(
        agg,
        x='Month-Year',
        y=['Credit', 'Debit', 'Net Flow'],
        barmode='group',
        title="Income, Spending, and Net Flow by Selected Months",
        labels={'value': 'Amount (J$)', 'Month-Year': 'Month'}
    )
    fig.update_layout(yaxis_tickprefix="J$")
    st.plotly_chart(fig, use_container_width=True)

elif period_type == "Quarterly (Current Quarter)":
    current_year = datetime.datetime.now().year
    current_quarter = get_current_quarter()

    st.markdown(f"**Showing data for Q{current_quarter} of {current_year}**")

    def quarter(month):
        return (month - 1) // 3 + 1

    filtered_trend = trend[(trend['Year'] == current_year) & (trend['Month'].apply(quarter) == current_quarter)]

    agg = filtered_trend.groupby(['Month', 'Category'])['Amount'].sum().unstack(fill_value=0)
    agg['Net Flow'] = agg.get('Credit', 0) - agg.get('Debit', 0)
    agg = agg.reset_index()
    agg['Month Name'] = agg['Month'].apply(lambda m: calendar.month_name[m])

    fig = px.bar(
        agg,
        x='Month Name',
        y=['Credit', 'Debit', 'Net Flow'],
        barmode='group',
        title=f"Income, Spending, and Net Flow for Q{current_quarter} {current_year}",
        labels={'value': 'Amount (J$)', 'Month Name': 'Month'}
    )
    fig.update_layout(yaxis_tickprefix="J$")
    st.plotly_chart(fig, use_container_width=True)

elif period_type == "Semi-Annually (Current Half-Year)":
    current_year = datetime.datetime.now().year
    current_half = get_current_half_year()

    half_label = "Jan - Jun" if current_half == 1 else "Jul - Dec"
    st.markdown(f"**Showing data for {half_label} {current_year}**")

    if current_half == 1:
        filtered_trend = trend[(trend['Year'] == current_year) & (trend['Month'].between(1, 6))]
    else:
        filtered_trend = trend[(trend['Year'] == current_year) & (trend['Month'].between(7, 12))]

    agg = filtered_trend.groupby(['Month', 'Category'])['Amount'].sum().unstack(fill_value=0)
    agg['Net Flow'] = agg.get('Credit', 0) - agg.get('Debit', 0)
    agg = agg.reset_index()
    agg['Month Name'] = agg['Month'].apply(lambda m: calendar.month_name[m])

    fig = px.bar(
        agg,
        x='Month Name',
        y=['Credit', 'Debit', 'Net Flow'],
        barmode='group',
        title=f"Income, Spending, and Net Flow for {half_label} {current_year}",
        labels={'value': 'Amount (J$)', 'Month Name': 'Month'}
    )
    fig.update_layout(yaxis_tickprefix="J$")
    st.plotly_chart(fig, use_container_width=True)


# -------------- Month Selection & Detailed Monthly Analysis --------------

data['Month'] = data['Date'].dt.strftime('%B')
available_months = sorted(data['Month'].unique(), key=lambda x: list(calendar.month_name).index(x))

month = st.selectbox("📅 Select Month to Explore", available_months, key="month_select")
filtered = data[data['Month'] == month].copy()

# Optional keyword search
search_keyword = st.text_input("Search in Descriptions (optional)")
if search_keyword:
    filtered = filtered[filtered['Description'].str.lower().str.contains(search_keyword.lower())]

# Classification
filtered['Spending Category'] = filtered['Description'].apply(
    lambda d: classify_expense(d, CATEGORY_KEYWORDS)
)

# Manual Tagging of Uncategorized
uncat = filtered[filtered['Spending Category'] == 'Uncategorized']
if not uncat.empty:
    st.subheader("🧩 Manually Tag Uncategorized Transactions")
    for i, row in uncat.iterrows():
        new_cat = st.selectbox(
            f"{row['Date'].date()} - {row['Description'][:40]}...",
            options=list(CATEGORY_KEYWORDS.keys()) + ["Other"],
            key=f"tag_{i}"
        )
        filtered.at[i, 'Spending Category'] = new_cat

# Show Transactions
st.subheader(f"📄 Transactions in {month}")
st.dataframe(filtered[['Date', 'Description', 'Amount', 'Category', 'Spending Category']])


# Spending Breakdown
st.subheader(f"📈 Spending Breakdown for {month}")
spend = filtered[filtered['Category'] == 'Debit']
summary = spend.groupby('Spending Category')['Amount'].sum().reset_index()
summary['Percentage'] = 100 * summary['Amount'] / summary['Amount'].sum()
st.dataframe(summary.style.format({"Amount": "J${:,.2f}", "Percentage": "{:.2f}%"}))

fig = px.pie(
    summary,
    names='Spending Category',
    values='Amount',
    title=f"{month} Spending Distribution",
    hole=0.4
)
st.plotly_chart(fig, use_container_width=True)


# Budget vs Actual
st.subheader(f"📏 Budget vs. Actual - {month}")
budget_df = pd.DataFrame.from_dict(MONTHLY_BUDGETS, orient='index', columns=['Budget']).reset_index()
budget_df.rename(columns={'index': 'Spending Category'}, inplace=True)
comparison = pd.merge(budget_df, summary, on='Spending Category', how='left')
comparison['Amount'] = comparison['Amount'].fillna(0)
comparison['Difference'] = comparison['Budget'] - comparison['Amount']
comparison['Status'] = comparison.apply(
    lambda row: "⚠️ Over Budget" if row['Amount'] > row['Budget'] else "Within Budget", axis=1
)

st.dataframe(
    comparison.style.format({"Budget": "J${:,.0f}", "Amount": "J${:,.0f}", "Difference": "J${:,.0f}"})
    .applymap(lambda v: 'color: red;' if isinstance(v, str) and 'Over' in v else '', subset=['Status'])
)

fig = px.bar(
    comparison,
    x='Spending Category',
    y=['Budget', 'Amount'],
    barmode='group',
    title="Budget vs. Actual Spending by Category",
    labels={"value": "J$", "variable": "Type"},
    text_auto=True
)
st.plotly_chart(fig, use_container_width=True)


# Savings Goal Check
st.subheader(f"🎯 Savings Goal Check for {month}")

total_income = filtered[filtered['Category'] == 'Credit']['Amount'].sum()
total_spending = filtered[filtered['Category'] == 'Debit']['Amount'].sum()
actual_savings = total_income - total_spending

st.markdown(f"**Savings Goal:** J${SAVINGS_GOAL:,.2f}")
st.markdown(f"**Actual Savings:** J${actual_savings:,.2f}")

if actual_savings >= SAVINGS_GOAL:
    st.success(f"🎉 Congrats! You've met your savings goal by J${actual_savings - SAVINGS_GOAL:,.2f}!")
else:
    st.warning(f"⚠️ You are J${SAVINGS_GOAL - actual_savings:,.2f} below your savings goal. Consider reviewing your budget.")


# Email alert trigger
if notify_email and sender_email and sender_password:
    overspent = comparison[comparison['Amount'] > comparison['Budget']]
    if not overspent.empty:
        subject = f"⚠️ Finance Tracker Alert: Overspending in {month}"
        body_lines = [f"Dear user,\n\nYou have overspent in the following categories for {month}:\n"]
        for _, row in overspent.iterrows():
            body_lines.append(f"- {row['Spending Category']}: Spent J${row['Amount']:.2f} (Budget: J${row['Budget']:.2f})")
        body_lines.append("\nPlease review your budget.")
        body = "\n".join(body_lines)

        if st.button("Send Overspending Alert Email"):
            sent = send_email_alert(
                receiver_email=notify_email,
                subject=subject,
                body=body,
                sender_email=sender_email,
                sender_password=sender_password,
                smtp_server=smtp_server,
                smtp_port=smtp_port
            )
            if sent:
                st.success("Email sent successfully!")
            else:
                st.error("Failed to send email. Check your credentials and internet connection.")


# Export Reports
st.subheader("📤 Export Reports")

report_text = f"Finance Report - {month}\n\nTransactions:\n"
for idx, row in filtered.iterrows():
    report_text += f"{row['Date'].date()} | {row['Description']} | J${row['Amount']:,.2f} | {row['Category']} | {row['Spending Category']}\n"

report_text += "\nSpending Summary:\n"
for idx, row in summary.iterrows():
    report_text += f"{row['Spending Category']}: J${row['Amount']:,.2f} ({row['Percentage']:.2f}%)\n"

report_text += "\nBudget vs Actual:\n"
for idx, row in comparison.iterrows():
    report_text += f"{row['Spending Category']}: Budget J${row['Budget']:,.2f}, Actual J${row['Amount']:,.2f}, Status: {row['Status']}\n"

export_format = st.selectbox("Select export format", options=["Excel", "PDF"])

if st.button("Download Report"):
    if export_format == "Excel":
        excel_bytes = export_to_excel(filtered)
        st.download_button(
            label="Download Excel File",
            data=excel_bytes,
            file_name=f"Finance_Report_{month}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        pdf_bytes = export_to_pdf(report_text)
        st.download_button(
            label="Download PDF File",
            data=pdf_bytes,
            file_name=f"Finance_Report_{month}.pdf",
            mime="application/pdf"
        )

