# A-Level Academic Pathway Recommendation System

### BTech Capstone Project | MUHORAKEYE MARTHA (25RP19281)
### G.S NYAKINAMA I TSS, Musanze, Rwanda | January 2026

## Project Overview

This Flask web application recommends the most suitable A-Level pathway for
O-Level students based on academic grades and student preferences.

The upload feature works locally only. It does not use any cloud AI API.
Digital PDFs are read with `pdfplumber`; scanned PDFs and images use local
Tesseract OCR when Tesseract and Poppler are installed.

## Project Structure

```text
alevel_system/
├── app.py
├── grade_extractor.py
├── requirements.txt
├── README.md
├── data/
│   └── survey_data.xlsx
├── model/
│   ├── train.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_cols.pkl
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── upload_review.html
│   └── about.html
└── uploads/
```

## How to Run

1. Create and activate a Python environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install Python dependencies.

```bash
pip install -r requirements.txt
```

3. Start the web server.

```bash
python app.py
```

4. Open the app in your browser.

```text
http://127.0.0.1:5000
```

## Upload/OCR Notes

Digital CAMIS-style PDFs can usually be read without extra system tools.

For scanned PDFs and image uploads, install these system tools:

- Tesseract OCR
- Poppler, for `pdf2image` PDF rendering

On Windows, make sure `tesseract.exe` and Poppler's `bin` folder are available
on the system `PATH`.

## Parent/Guardian Email Notifications

The student information form includes an optional parent/guardian email field.
When a recommendation result is generated, the app can send a "result ready to
print" notification if SMTP is configured.

You can configure this in either of two ways:

1. Copy `email_settings.example.env` to `email_settings.env` and fill in the
   real sender account details.
2. Or set these environment variables before starting the server:

```text
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=school.email@example.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=school.email@example.com
SMTP_USE_TLS=true
APP_PUBLIC_URL=http://127.0.0.1:5000
```

`APP_PUBLIC_URL` is optional and is only included in the email body when set.
Use an app password or school SMTP account rather than placing a personal
password in the source code.

For Gmail, use a Gmail App Password, not your normal Gmail password. After
editing `email_settings.env`, restart the Flask server and test with:

```bash
python test_email_settings.py parent@example.com
```

## Training

The trained model files are already included. Re-train only when
`data/survey_data.xlsx` changes:

```bash
python model/train.py
```

## Algorithms Used

- Random Forest Classifier
- MinMax Scaling
- Stratified K-Fold Cross-Validation
- Feature engineering from subject groups

## Features

- Student grade input form
- Top A-Level pathway recommendations with confidence percentages
- Student preference inputs
- Report-card upload and local grade extraction
- Analytics dashboard
- Print-friendly results page
- Responsive Bootstrap interface

## Tech Stack

| Component | Technology |
| --- | --- |
| Backend | Python, Flask |
| ML | scikit-learn, pandas, NumPy |
| OCR | pdfplumber, Tesseract OCR, pdf2image |
| Frontend | HTML, Bootstrap 5, Chart.js |
| Data | Excel |

Supervisor: UWIZEYE Samuel | Co-Supervisor: MBABAZI Mary
