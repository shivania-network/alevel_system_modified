"""
=============================================================================
 app.py  —  Flask Web Application: A-Level Pathway Recommendation System
 BTech Capstone Project — MUHORAKEYE MARTHA (25RP19281)
 G.S NYAKINAMA I TSS, Musanze, Northern Province, Rwanda

=============================================================================

PURPOSE:
  Main web server:
    /           : Manual grade entry form
    /predict    : Process form → pathway recommendation
    /dashboard  : Analytics dashboard
    /upload     : Upload O-Level report cards
    /upload/process : OCR extraction + review page
    /about      : About page

GRADING SCALE (CAMIS 2022+, confirmed from real report cards):
  A = 80-100%,  B = 75-79%,  C = 70-74%,  D = 65-69%,
  E = 60-64%,  S = 50-59%  (Satisfactory),  F < 50%
=============================================================================
"""

from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from html import escape
import os, json, uuid, warnings, smtplib
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from grade_extractor import (extract_text_from_file, extract_grades_from_text,
                             average_grades, average_percentages,
                             extract_name_from_text, names_match,
                             extract_class_from_text, extract_gender_from_text,
                             extract_school_from_text, genders_match,
                             schools_match, looks_like_report_card_text)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'alevel-recommender-2026')

BASE          = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE, 'uploads')
ALLOWED_EXT   = {'pdf', 'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def load_email_settings():
    """
    Load optional local SMTP settings from email_settings.env.
    Existing environment variables take priority.
    """
    settings_path = os.path.join(BASE, 'email_settings.env')
    if not os.path.exists(settings_path):
        return
    try:
        with open(settings_path, encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                current = os.environ.get(key, '').strip()
                if key and (not current or current == 'PASTE_GMAIL_APP_PASSWORD_HERE'):
                    os.environ[key] = value
    except Exception:
        pass


load_email_settings()


def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.context_processor
def inject_kiro_modal():
    """Pop one-shot modal payload set by redirect-after-validation."""
    return {'kiro_modal': session.pop('kiro_modal', None)}


def _file_uploaded(key):
    f = request.files.get(key)
    return bool(f and f.filename and allowed_file(f.filename))


def collect_invalid_marks(data):
    """Return list of human-readable invalid mark entries (empty if all OK)."""
    invalid = []
    for sub in SUBJECTS:
        for yr in ('s1', 's2', 's3'):
            raw = data.get(f'{yr}_{sub}', '').strip()
            if not raw:
                continue
            try:
                v = float(raw)
                if v < 0 or v > 100:
                    invalid.append(f'{sub} ({yr.upper()}: {raw})')
            except ValueError:
                invalid.append(f'{sub} ({yr.upper()}: not a number)')
    for sub in SUBJECTS:
        raw = data.get(sub, '').strip()
        if not raw:
            continue
        try:
            v = float(raw)
            if v < 0 or v > 100:
                invalid.append(f'{sub} (average: {raw})')
        except ValueError:
            pass  # letter grade from upload review is OK
    return invalid


def redirect_with_modal(modal, fallback_endpoint='grades'):
    session['kiro_modal'] = modal
    return redirect(request.referrer or url_for(fallback_endpoint))


def get_guardian_email(data):
    """Return the optional parent/guardian email entered on the form."""
    return (data.get('guardian_email') or '').strip()


def is_valid_email(addr):
    """Validate one parent/guardian email address."""
    parsed = parseaddr(addr)[1]
    if not parsed or parsed != addr or len(parsed) > 254 or ' ' in parsed:
        return False
    local, sep, domain = parsed.rpartition('@')
    return bool(local and sep and '.' in domain and
                not domain.startswith('.') and not domain.endswith('.'))


def subject_has_entered_mark(data, subject):
    """True when a subject mark was typed/extracted, not auto-filled as blank."""
    for yr in ('s1', 's2', 's3'):
        if data.get(f'{yr}_{subject}', '').strip():
            return True

    raw = data.get(subject, '').strip()
    if not raw:
        return False
    if raw.upper() in GRADE_MAP:
        return True
    try:
        return float(raw) > 0
    except ValueError:
        return False


def build_result_email_html(student_name, results, grades, student_choices,
                            trace_id=None):
    """Build a printable HTML version of the recommendation result email."""
    results = results or []
    grades = grades or {}
    student_choices = student_choices or {}
    top = results[0] if results else {}

    result_rows = ''.join(
        '<tr>'
        f'<td>{escape(str(item.get("rank", "-")))}</td>'
        f'<td>{escape(str(item.get("pathway", "N/A")))}</td>'
        f'<td>{escape(str(item.get("probability", "N/A")))}%</td>'
        '</tr>'
        for item in results
    ) or '<tr><td colspan="3">No pathway result available.</td></tr>'

    grade_rows = ''.join(
        '<tr>'
        f'<td>{escape(str(subject))}</td>'
        f'<td>{escape(str(grade))}</td>'
        '</tr>'
        for subject, grade in grades.items()
    ) or '<tr><td colspan="2">No grades available.</td></tr>'

    choices = {
        'Wish / Passion': student_choices.get('wish', '-'),
        'Career Interest': student_choices.get('career', '-'),
        'Strongest Talent': student_choices.get('talent', '-'),
        'Labor Market': student_choices.get('labor', '-'),
    }
    choice_rows = ''.join(
        '<tr>'
        f'<td>{escape(label)}</td>'
        f'<td>{escape(str(value))}</td>'
        '</tr>'
        for label, value in choices.items()
    )

    trace_line = (
        f'<br>Message reference: <strong>{escape(trace_id)}</strong>'
        if trace_id else ''
    )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>A-Level pathway result</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #111; line-height: 1.45; }}
    .page {{ max-width: 760px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 22px; margin: 0 0 4px; }}
    h2 {{ font-size: 16px; margin: 24px 0 8px; }}
    .muted {{ color: #555; font-size: 13px; margin: 0 0 18px; }}
    .top {{ border: 2px solid #111; padding: 14px; margin: 16px 0; }}
    .top strong {{ font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #111; padding: 8px; font-size: 13px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    .footer {{ margin-top: 24px; font-size: 12px; color: #444; }}
    @media print {{
      body {{ margin: 0; }}
      .page {{ max-width: none; padding: 18px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>A-Level Pathway Recommendation Result</h1>
    <p class="muted">Printable result for {escape(student_name)}</p>

    <div class="top">
      <div>Top Recommendation</div>
      <strong>{escape(str(top.get('pathway', 'N/A')))}</strong>
      <div>{escape(str(top.get('probability', 'N/A')))}% confidence</div>
    </div>

    <h2>Ranked Pathway Results</h2>
    <table>
      <thead><tr><th>Rank</th><th>Pathway</th><th>Confidence</th></tr></thead>
      <tbody>{result_rows}</tbody>
    </table>

    <h2>Student Preferences</h2>
    <table>
      <tbody>{choice_rows}</tbody>
    </table>

    <h2>O-Level Grades Used</h2>
    <table>
      <thead><tr><th>Subject</th><th>Grade / Percentage</th></tr></thead>
      <tbody>{grade_rows}</tbody>
    </table>

    <p class="footer">
      This is an automatic printable result from the A-Level Academic Pathway
      Recommendation System. Open this email and use Print to keep a paper copy.
      {trace_line}
    </p>
  </div>
</body>
</html>"""


def send_result_ready_email(guardian_email, student_name, results=None,
                            grades=None, student_choices=None):
    """
    Email a parent/guardian the recommendation result.
    SMTP is configured through environment variables so secrets are never stored
    in the source code.
    """
    if not guardian_email:
        return None

    load_email_settings()

    host = os.environ.get('SMTP_HOST', '').strip()
    sender = (os.environ.get('SMTP_FROM') or
              os.environ.get('SMTP_USERNAME') or '').strip()
    if not host or not sender:
        return {
            'status': 'not_configured',
            'message': 'Result email is ready, but SMTP is not configured yet.',
        }

    port = int(os.environ.get('SMTP_PORT', '587'))
    username = os.environ.get('SMTP_USERNAME', '').strip()
    raw_password = os.environ.get('SMTP_PASSWORD', '')
    password = ''.join(raw_password.split())
    if username and (not password or password == 'PASTE_GMAIL_APP_PASSWORD_HERE'):
        return {
            'status': 'not_configured',
            'message': 'Result email is ready, but the Gmail app password is not configured yet.',
        }
    use_ssl = os.environ.get('SMTP_USE_SSL', '').lower() in ('1', 'true', 'yes')
    use_tls = os.environ.get('SMTP_USE_TLS', 'true').lower() not in ('0', 'false', 'no')
    app_url = os.environ.get('APP_PUBLIC_URL', '').strip()

    trace_id = uuid.uuid4().hex[:10].upper()
    email_subject = f'A-Level pathway result for {student_name} [{trace_id}]'
    results = results or []
    grades = grades or {}
    student_choices = student_choices or {}

    body_lines = [
        'Dear Parent/Guardian,',
        '',
        f'The A-Level pathway recommendation result for {student_name} is ready.',
        f'Message reference: {trace_id}',
        '',
    ]

    if guardian_email.lower() == sender.lower():
        body_lines.extend([
            'Note: this message was sent from the same Gmail account to itself.',
            'If it is not in the Inbox, check Sent, All Mail, Spam, or search Gmail for the message reference above.',
            '',
        ])

    if results:
        top = results[0]
        body_lines.extend([
            'Top recommendation:',
            f"- {top.get('pathway', 'N/A')} ({top.get('probability', 'N/A')}% confidence)",
            '',
            'Ranked pathway results:',
        ])
        for item in results:
            body_lines.append(
                f"{item.get('rank', '-')}. {item.get('pathway', 'N/A')} - "
                f"{item.get('probability', 'N/A')}% confidence"
            )
        body_lines.append('')

    if student_choices:
        body_lines.extend([
            'Student preferences:',
            f"- Wish / passion: {student_choices.get('wish', '-')}",
            f"- Career interest: {student_choices.get('career', '-')}",
            f"- Strongest talent: {student_choices.get('talent', '-')}",
            f"- Labor market: {student_choices.get('labor', '-')}",
            '',
        ])

    if grades:
        body_lines.append('O-Level grades used:')
        for subject_name, grade in grades.items():
            body_lines.append(f"- {subject_name}: {grade}")
        body_lines.append('')

    body_lines.append('Please contact the school office if you need the printed copy.')

    if app_url:
        body_lines.extend(['', f'System link: {app_url}'])
    body_lines.extend([
        '',
        'This is an automatic printable result from the A-Level Academic Pathway Recommendation System.',
    ])

    msg = EmailMessage()
    msg['Subject'] = email_subject
    msg['From'] = sender
    msg['To'] = guardian_email
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid()
    msg['X-A-Level-Result-Reference'] = trace_id
    msg.set_content('\n'.join(body_lines))
    msg.add_alternative(
        build_result_email_html(student_name, results, grades, student_choices,
                                trace_id=trace_id),
        subtype='html')

    try:
        if use_ssl:
            smtp = smtplib.SMTP_SSL(host, port, timeout=10)
        else:
            smtp = smtplib.SMTP(host, port, timeout=10)
        with smtp:
            if use_tls and not use_ssl:
                smtp.starttls()
            if username:
                smtp.login(username, password)
            refused = smtp.send_message(msg)
        if refused:
            refused_to = ', '.join(refused.keys())
            return {
                'status': 'failed',
                'message': f'Gmail refused delivery to: {refused_to}.',
            }
        return {
            'status': 'sent',
            'message': 'Message sent to the parent/guardian.',
        }
    except smtplib.SMTPAuthenticationError:
        return {
            'status': 'failed',
            'message': 'Gmail rejected the sender login. Check the Gmail app password and 2-Step Verification.',
        }
    except smtplib.SMTPRecipientsRefused:
        return {
            'status': 'failed',
            'message': 'The recipient email address was refused by the mail server.',
        }
    except smtplib.SMTPException:
        return {
            'status': 'failed',
            'message': 'The mail server rejected the email request. Check SMTP settings.',
        }
    except OSError:
        return {
            'status': 'failed',
            'message': 'Could not connect to the Gmail SMTP server. Check internet connection or firewall.',
        }


# ── Load ML model artifacts ───────────────────────────────────────────────
model        = joblib.load(os.path.join(BASE, 'model', 'model.pkl'))
scaler       = joblib.load(os.path.join(BASE, 'model', 'scaler.pkl'))
le           = joblib.load(os.path.join(BASE, 'model', 'label_encoder.pkl'))
feature_cols = joblib.load(os.path.join(BASE, 'model', 'feature_cols.pkl'))

# ── Constants ─────────────────────────────────────────────────────────────
SUBJECTS = ['Math', 'Physics', 'Chemistry', 'Biology', 'Geography',
            'History', 'Entrepreneurship', 'English', 'Kinyarwanda',
            'French', 'Kiswahili']

# S = Satisfactory (50-59%) — used by all CAMIS schools
GRADES    = ['A', 'B', 'C', 'D', 'E', 'S', 'F']
GRADE_MAP = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'S': 2, 'F': 1}

def percent_to_letter(pct):
    """
    CAMIS/modern Rwanda grading scale (confirmed from images 3–6):
    A≥80, B≥75, C≥70, D≥65, E≥60, S≥50, F<50
    """
    if pct >= 80: return 'A'
    if pct >= 75: return 'B'
    if pct >= 70: return 'C'
    if pct >= 65: return 'D'
    if pct >= 60: return 'E'
    if pct >= 50: return 'S'
    return 'F'

def percent_to_grade_value(pct):
    return GRADE_MAP[percent_to_letter(pct)]

# ── Student preference options ────────────────────────────────────────────
WISH_OPTIONS = [
    ('Sciences',             '🔬 Sciences'),
    ('Arts & Humanities',    '📚 Arts & Humanities'),
    ('Languages',            '🗣️ Languages'),
    ('Technology',           '💻 Technology'),
    ('Business & Economics', '📊 Business & Economics'),
]
CAREER_OPTIONS = [
    ('Health & Medicine',        '🏥 Health & Medicine'),
    ('Engineering & Technology', '⚙️ Engineering & Technology'),
    ('Business & Finance',       '💼 Business & Finance'),
    ('Law & Policy',             '⚖️ Law & Policy'),
    ('Journalism & Diplomacy',   '📰 Journalism & Diplomacy'),
    ('Education & Teaching',     '📖 Education & Teaching'),
]
TALENT_OPTIONS = [
    ('Analytical / Problem-solving',    '🧠 Analytical / Problem-solving'),
    ('Creative / Artistic',             '🎨 Creative / Artistic'),
    ('Communication / Public Speaking', '🎤 Communication / Public Speaking'),
    ('Technical / Hands-on',            '🔧 Technical / Hands-on'),
    ('Reading, Writing & Research',     '✍️ Reading, Writing & Research'),
]
LABOR_OPTIONS = [
    ('Yes',      '✅ Yes — job demand matters to me'),
    ('No',       '❌ No — I follow my passion'),
    ('Not Sure', '🤔 Not Sure — I need guidance'),
]

# ── Encoding maps ─────────────────────────────────────────────────────────
WISH_MAP = {
    'Sciences': 1, 'Arts & Humanities': 2, 'Languages': 3,
    'Technology': 4, 'Business & Economics': 5
}
CAREER_MAP = {
    'Health & Medicine': 1, 'Engineering & Technology': 2,
    'Business & Finance': 3, 'Law & Policy': 4,
    'Journalism & Diplomacy': 5, 'Education & Teaching': 6
}
TALENT_MAP = {
    'Analytical / Problem-solving': 1, 'Creative / Artistic': 2,
    'Communication / Public Speaking': 3, 'Technical / Hands-on': 4,
    'Reading, Writing & Research': 5
}
LABOR_MAP = {'Yes': 1, 'No': 0, 'Not Sure': 2}

# ── Preference → Pathway alignment map ───────────────────────────────────
# Defines how strongly each wish/career/talent selection should boost
# each pathway's probability.  Values are additive boosts (0.0–0.25).
# This ensures student preferences materially influence the final ranking,
# especially when grades are average across all subjects.
PREFERENCE_BOOST = {
    # (wish, career, talent) → {pathway: boost}
    'wish': {
        'Sciences':             {'Mathematics & Sciences': 0.18},
        'Arts & Humanities':    {'Arts & Humanities': 0.18},
        'Languages':            {'Languages': 0.18},
        'Technology':           {'TVET — Technology': 0.18, 'Mathematics & Sciences': 0.06},
        'Business & Economics': {'Mathematics & Sciences': 0.10, 'Arts & Humanities': 0.08},
    },
    'career': {
        'Health & Medicine':        {'Mathematics & Sciences': 0.12},
        'Engineering & Technology': {'Mathematics & Sciences': 0.10, 'TVET — Technology': 0.10},
        'Business & Finance':       {'Mathematics & Sciences': 0.08, 'Arts & Humanities': 0.06},
        'Law & Policy':             {'Arts & Humanities': 0.12},
        'Journalism & Diplomacy':   {'Languages': 0.12, 'Arts & Humanities': 0.06},
        'Education & Teaching':     {'Languages': 0.08, 'Arts & Humanities': 0.08},
    },
    'talent': {
        'Analytical / Problem-solving': {'Mathematics & Sciences': 0.10},
        'Creative / Artistic':          {'Arts & Humanities': 0.10},
        'Communication / Public Speaking': {'Languages': 0.10, 'Arts & Humanities': 0.06},
        'Technical / Hands-on':         {'TVET — Technology': 0.10, 'Mathematics & Sciences': 0.06},
        'Reading, Writing & Research':  {'Arts & Humanities': 0.10, 'Languages': 0.06},
    },
}

# How much preferences adjust the final ranking (rest comes from marks / ML model)
PREFERENCE_BLEND_WEIGHT = 0.30


def collect_missing_fields(data):
    """Return list of human-readable missing required form fields."""
    missing = []
    if not data.get('student_name', '').strip():
        missing.append('Student full name')
    if not get_guardian_email(data):
        missing.append('Parent/Guardian email')
    if data.get('gender', '') not in ('0', '1'):
        missing.append('Gender')
    if data.get('student_wish', '') not in WISH_MAP:
        missing.append('Study area you wish to pursue')
    if data.get('career_interest', '') not in CAREER_MAP:
        missing.append('Career interest')
    if data.get('talent', '') not in TALENT_MAP:
        missing.append('Strongest talent')
    if data.get('labor_market', '') not in LABOR_MAP:
        missing.append('Labor market preference')
    return missing


def parse_gender(data):
    raw = data.get('gender', '')
    if raw in ('0', '1'):
        return int(raw)
    return None


def build_preference_distribution(wish, career, talent, pathways):
    """Normalised preference scores across pathways (from wish + career + talent)."""
    scores = {p: 0.0 for p in pathways}
    for dim, val in [('wish', wish), ('career', career), ('talent', talent)]:
        for pathway, amount in PREFERENCE_BOOST.get(dim, {}).get(val, {}).items():
            if pathway in scores:
                scores[pathway] += amount
    total = sum(scores.values())
    if total <= 0:
        n = len(pathways) or 1
        return {p: 1.0 / n for p in pathways}
    return {k: v / total for k, v in scores.items()}


def blend_grades_and_preferences(model_proba, wish, career, talent):
    """
    Combine ML probabilities (marks + engineered scores) with student preferences.
    Default: 70% academic marks, 30% personal preferences.
    """
    pathways = list(model_proba.keys())
    pref = build_preference_distribution(wish, career, talent, pathways)
    w = PREFERENCE_BLEND_WEIGHT
    combined = {}
    for p in pathways:
        combined[p] = (1.0 - w) * model_proba.get(p, 0.0) + w * pref.get(p, 0.0)
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}
    return combined, pref


def apply_preference_boost(proba_dict, wish, career, talent):
    """Blend marks-based model output with preference alignment."""
    combined, _pref = blend_grades_and_preferences(proba_dict, wish, career, talent)
    return combined


# ── Pathway info ──────────────────────────────────────────────────────────
PATHWAY_INFO = {
    'Mathematics & Sciences': {
        'color':    '#1e3a5f', 'icon': '🔬',
        'subjects': 'Mathematics, Physics, Chemistry, Biology, Geography, Economics',
        'mandatory':'Mathematics, English, ICT, Entrepreneurship, General Studies, P.E',
        'careers':  'Doctor, Pharmacist, Civil Engineer, Software Developer, '
                    'Economist, Financial Analyst, Environmental Scientist, Data Analyst',
        'description': ('Covers physical sciences, life sciences and quantitative social sciences. '
                        'Designed for students with strong Mathematics and Sciences skills '
                        'aiming for careers in health, engineering, business or technology. '
                        'Replaces the old PCM, PCB, MCB, MEG, MPG and MCE combinations.'),
        'category': 'Mathematics & Sciences',
    },
    'Arts & Humanities': {
        'color':    '#be185d', 'icon': '📚',
        'subjects': 'History, Geography, Literature in English, Psychology, Economics',
        'mandatory':'Mathematics, English, ICT, Entrepreneurship, General Studies, P.E',
        'careers':  'Lawyer, Psychologist, Historian, Social Worker, Policy Analyst, '
                    'Diplomat, Journalist, Community Development Officer',
        'description': ('Develops critical thinking in social sciences, humanities and creative arts. '
                        'Best suited for students interested in law, governance, education, '
                        'social sciences and public administration. '
                        'Replaces the old HEG, HEL and HGL combinations.'),
        'category': 'Arts & Humanities',
    },
    'Languages': {
        'color':    '#16a34a', 'icon': '🗣️',
        'subjects': 'English, French, Kinyarwanda, Kiswahili',
        'mandatory':'Mathematics, English, ICT, Entrepreneurship, General Studies, P.E',
        'careers':  'Journalist, Diplomat, Translator, Tourism Professional, '
                    'Writer, Language Teacher, International Relations Officer',
        'description': ('Focuses on multilingual communication in English, French, '
                        'Kinyarwanda and Kiswahili. Ideal for journalism, diplomacy, '
                        'translation, tourism and international relations. '
                        'Replaces the old EFK, LFK and EKK combinations.'),
        'category': 'Languages',
    },
    'TVET — Technology': {
        'color':    '#0f766e', 'icon': '💻',
        'subjects': 'Computer Application, Electronics & Telecommunication, ICT Systems',
        'mandatory':'Mathematics, English, ICT, Entrepreneurship, General Studies, P.E',
        'careers':  'Software Developer, Network Engineer, Electronics Technician, '
                    'ICT Support Specialist, Database Administrator',
        'description': ('Technical and Vocational Education and Training pathway. '
                        'Provides hands-on practical skills in computer systems, electronics '
                        'and ICT directly applicable to the job market. UNCHANGED by the 2025 reform.'),
        'category': 'TVET',
    },
}

def get_pathway_info(name):
    for k, v in PATHWAY_INFO.items():
        if k.lower().strip() == name.lower().strip():
            return v
    return {'color':'#4b5563','icon':'🎓','subjects':'Various','mandatory':'',
            'careers':'Various','description':'A broad academic pathway.','category':'General'}


# ── Report slots ──────────────────────────────────────────────────────────
REPORT_SLOTS = [
    ('s1',   'S1 — Full Report (All 3 Terms)'),
    ('s2',   'S2 — Full Report (All 3 Terms)'),
    ('s3t1', 'S3 — Term 1'),
    ('s3t2', 'S3 — Term 2'),
]
REPORT_SLOT_SHORT_LABELS = {
    's1':   'S1 full report',
    's2':   'S2 full report',
    's3t1': 'S3 term 1',
    's3t2': 'S3 term 2',
}
REPORT_GROUPS = [
    ('Senior 1 (S1)', 'Upload one report containing all 3 terms', 'single',
     [('s1', 'S1 — Full Report')]),
    ('Senior 2 (S2)', 'Upload one report containing all 3 terms', 'single',
     [('s2', 'S2 — Full Report')]),
    ('Senior 3 (S3)', 'Term 1 & Term 2 only — Term 3 not available (exam period)', 'partial',
     [('s3t1', 'S3 — Term 1'), ('s3t2', 'S3 — Term 2')]),
]


def short_report_label(slot_key, fallback='Report'):
    return REPORT_SLOT_SHORT_LABELS.get(slot_key, fallback)


def report_names_conflict(report_names):
    """True when extracted report names do not all point to the same student."""
    for i, left in enumerate(report_names):
        for right in report_names[i + 1:]:
            if not (names_match(left['name'], right['name']) and
                    names_match(right['name'], left['name'])):
                return True
    return False


# ════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/grades')
def grades():
    return render_template('enter_grades.html',
        subjects=SUBJECTS, grades=GRADES,
        wish_options=WISH_OPTIONS, career_options=CAREER_OPTIONS,
        talent_options=TALENT_OPTIONS, labor_options=LABOR_OPTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Build feature vector from form data, run Random Forest, apply
    preference boost, and return ranked pathway results.

    Grade input modes:
      1. 3-year form  — hidden fields contain pre-averaged percentages
                        computed by JavaScript from S1/S2/S3 inputs.
      2. Upload review — letter grades (A-F/S) from OCR extraction.
    """
    data = request.form

    missing = collect_missing_fields(data)
    if missing:
        return redirect_with_modal({
            'type': 'warning',
            'title': 'Please Complete All Required Fields',
            'subtitle': 'Some information is missing',
            'body': 'Fill in every required field below before getting your result:',
            'items': missing,
            'okText': 'Go Back & Fill',
        })

    guardian_email = get_guardian_email(data)
    if not is_valid_email(guardian_email):
        return redirect_with_modal({
            'type': 'warning',
            'title': 'Invalid Parent/Guardian Email',
            'subtitle': 'Please check the email address',
            'body': 'Enter a valid parent or guardian email address before getting the result.',
            'okText': 'Go Back & Fix',
        })

    gender = parse_gender(data)
    if gender is None:
        return redirect_with_modal({
            'type': 'warning',
            'title': 'Gender Required',
            'subtitle': 'Please select your gender',
            'body': 'Choose Male or Female in the student information section.',
            'okText': 'Go Back & Fill',
        })

    invalid_marks = collect_invalid_marks(data)
    if invalid_marks:
        return redirect_with_modal({
            'type': 'danger',
            'title': 'Invalid Marks Entered',
            'subtitle': 'Marks must be between 0 and 100',
            'body': 'The following entries have invalid values. Please enter real percentage marks (0–100):',
            'items': invalid_marks,
            'okText': 'Go Back & Fix',
        })

    try:
        return _predict_inner(data, gender)
    except Exception:
        return redirect_with_modal({
            'type': 'danger',
            'title': 'Something Went Wrong',
            'subtitle': 'Could not generate your recommendation',
            'body': 'Please check that all marks and preference fields are filled in correctly, then try again.',
            'okText': 'Go Back & Fix',
        }, fallback_endpoint='grades')


def _predict_inner(data, gender):
    # Read grades — hidden fields hold the JS-computed 3-year average (%)
    # or letter grades when coming from the upload review form.
    grade_vals     = []
    grades_display = {}
    raw_pcts       = []

    for sub in SUBJECTS:
        raw = data.get(sub, '').strip()
        try:
            pct = float(raw)
            gv  = percent_to_grade_value(pct)
            grades_display[sub] = f"{percent_to_letter(pct)} ({int(round(pct))}%)"
            raw_pcts.append(pct)
        except (ValueError, TypeError):
            letter = raw.upper()
            gv = GRADE_MAP.get(letter, 3)
            grades_display[sub] = letter if letter in GRADE_MAP else '—'
            letter_to_pct = {'A': 85, 'B': 77, 'C': 72, 'D': 67,
                             'E': 62, 'S': 55, 'F': 25}
            raw_pcts.append(letter_to_pct.get(letter, 0))
        grade_vals.append(gv)

    # ── Eligibility gate — only block if ALL subjects are zero ──────────
    # Every student deserves a recommendation based on their best subjects.
    # The only case we block is when every single subject is 0% (no data).
    all_zero = all(p == 0.0 for p in raw_pcts)

    if all_zero:
        student_name = data.get('student_name', 'Student').strip() or 'Student'
        return render_template('not_eligible.html',
            student_name=student_name,
            passed_count=0,
            failed_subjects=SUBJECTS,
            grades=grades_display,
            subjects=SUBJECTS,
            not_eligible_confidence=100.0)

    raw_wish   = data.get('student_wish', '').strip()
    raw_career = data.get('career_interest', '').strip()
    raw_talent = data.get('talent', '').strip()
    raw_labor  = data.get('labor_market', 'Not Sure').strip()

    student_wish    = WISH_MAP[raw_wish]
    career_interest = CAREER_MAP[raw_career]
    talent          = TALENT_MAP[raw_talent]
    labor_market    = LABOR_MAP.get(raw_labor, 2)

    # Engineered group scores (same formulas as train.py)
    g = dict(zip(SUBJECTS, grade_vals))
    science_score    = np.mean([g['Math'], g['Physics'], g['Chemistry'], g['Biology']])
    humanities_score = np.mean([g['History'], g['Geography'], g['English']])
    language_score   = np.mean([g['English'], g['Kinyarwanda'], g['French'], g['Kiswahili']])
    tech_score       = np.mean([g['Math'], g['Physics'], g['Entrepreneurship']])
    business_score   = np.mean([g['Math'], g['Geography'], g['Entrepreneurship']])
    overall_avg      = np.mean(grade_vals)

    features = np.array([[gender] + grade_vals +
                          [student_wish, career_interest, talent, labor_market] +
                          [science_score, humanities_score, language_score,
                           tech_score, business_score, overall_avg]], dtype=float)

    features_scaled = scaler.transform(features)
    proba_raw       = model.predict_proba(features_scaled)[0]

    # Build probability dict keyed by pathway name
    proba_dict = {le.classes_[i]: float(proba_raw[i])
                  for i in range(len(le.classes_))}

    # Remove "Not Eligible" from the pathway ranking — eligibility is
    # handled by the hard gate above (all-zeros check). Any student who
    # reaches this point deserves a ranked recommendation.
    proba_dict.pop('Not Eligible', None)

    # Blend marks (ML model) with preferences — 70% grades, 30% wish/career/talent
    proba_boosted, _pref_dist = blend_grades_and_preferences(
        proba_dict, raw_wish, raw_career, raw_talent)

    # Rank pathways
    ranked = sorted(proba_boosted.items(), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (pathway, prob) in enumerate(ranked, 1):
        info = get_pathway_info(pathway)
        model_pct = round(proba_dict.get(pathway, 0) * 100, 1)
        results.append({
            'rank':        rank,
            'pathway':     pathway,
            'probability': round(prob * 100, 1),
            'marks_score': model_pct,
            'color':       info['color'],
            'icon':        info['icon'],
            'subjects':    info['subjects'],
            'mandatory':   info['mandatory'],
            'careers':     info['careers'],
            'description': info['description'],
            'category':    info['category'],
        })

    student_name    = data.get('student_name', 'Student').strip() or 'Student'
    guardian_email  = get_guardian_email(data)
    student_choices = {
        'wish':   raw_wish   or '—',
        'career': raw_career or '—',
        'talent': raw_talent or '—',
        'labor':  raw_labor  or '—',
    }

    # Collect per-year marks for display on result page
    # Form sends s1_Math, s2_Math, s3_Math etc. as hidden fields
    year_marks = {}
    for sub in SUBJECTS:
        year_marks[sub] = {}
        for yr in ['s1', 's2', 's3']:
            v = data.get(f'{yr}_{sub}', '').strip()
            if v:
                try:
                    val = float(v)
                    if val > 0:   # only store if actually entered
                        year_marks[sub][yr] = val
                except ValueError:
                    pass

    # Pass all form data as hidden fields so student can change preferences
    # and resubmit without re-entering grades
    form_data = {k: v for k, v in data.items()}
    email_grades = {
        sub: grades_display[sub]
        for sub in SUBJECTS
        if subject_has_entered_mark(data, sub)
    }
    email_notice = send_result_ready_email(
        guardian_email, student_name,
        results=results,
        grades=email_grades,
        student_choices=student_choices)

    return render_template('result.html',
        student_name=student_name,
        guardian_email=guardian_email,
        email_notice=email_notice,
        gender_label='Male' if gender == 1 else 'Female',
        gender_val=str(gender),
        grades=grades_display,
        results=results,
        subjects=SUBJECTS,
        student_choices=student_choices,
        year_marks=year_marks,
        marks_weight=int((1 - PREFERENCE_BLEND_WEIGHT) * 100),
        prefs_weight=int(PREFERENCE_BLEND_WEIGHT * 100),
        form_data=form_data,
        wish_options=WISH_OPTIONS,
        career_options=CAREER_OPTIONS,
        talent_options=TALENT_OPTIONS,
        labor_options=LABOR_OPTIONS)


@app.route('/upload', methods=['GET'])
def upload():
    return render_template('upload.html',
        report_groups=REPORT_GROUPS,
        subjects=SUBJECTS, grades=GRADES,
        wish_options=WISH_OPTIONS, career_options=CAREER_OPTIONS,
        talent_options=TALENT_OPTIONS, labor_options=LABOR_OPTIONS)


@app.route('/upload/process', methods=['POST'])
def upload_process():
    """
    Receive uploaded report cards, extract grades locally, average across
    reports, and show the review page.
    """
    student_name = request.form.get('student_name', '').strip()
    gender_raw   = request.form.get('gender', '')
    school_name  = request.form.get('school', '').strip()
    guardian_email = get_guardian_email(request.form)

    upload_missing = []
    if not student_name:
        upload_missing.append('Student full name')
    if not guardian_email:
        upload_missing.append('Parent/Guardian email')
    if gender_raw not in ('0', '1'):
        upload_missing.append('Gender')
    if upload_missing:
        return redirect_with_modal({
            'type': 'warning',
            'title': 'Please Complete Student Information',
            'subtitle': 'Name and gender are required',
            'body': 'Fill in the following before uploading reports:',
            'items': upload_missing,
            'okText': 'Go Back & Fill',
        }, fallback_endpoint='upload')

    if not is_valid_email(guardian_email):
        return redirect_with_modal({
            'type': 'warning',
            'title': 'Invalid Parent/Guardian Email',
            'subtitle': 'Please check the email address',
            'body': 'Enter a valid parent or guardian email address before uploading reports.',
            'okText': 'Go Back & Fix',
        }, fallback_endpoint='upload')

    student_name = student_name or 'Student'
    gender       = gender_raw

    missing_reports = [
        short_report_label(key, label) for key, label in REPORT_SLOTS
        if not _file_uploaded(key)
    ]
    if missing_reports:
        return redirect_with_modal({
            'type': 'warning',
            'title': 'Missing Reports',
            'subtitle': 'Upload all 4 required files',
            'body': 'These report cards are still missing:',
            'items': missing_reports,
            'okText': 'Go Back & Upload',
        }, fallback_endpoint='upload')

    all_grades    = []
    uploaded_info = []
    invalid_reports    = []   # list of {slot, file}
    student_mismatches = []   # list of {slot, file, field, entered, found}
    slot_mismatches    = []   # list of {slot, file, expected_class, found_class}
    report_names       = []   # list of {slot_key, slot, file, name}

    # Which year-level each upload slot expects
    SLOT_EXPECTED_CLASS = {
        's1':   'S1',
        's2':   'S2',
        's3t1': 'S3',
        's3t2': 'S3',
    }

    for slot_key, slot_label in REPORT_SLOTS:
        f = request.files.get(slot_key)

        if f and f.filename and allowed_file(f.filename):
            ext      = f.filename.rsplit('.', 1)[1].lower()
            filename = secure_filename(f'{slot_key}_{uuid.uuid4().hex[:8]}.{ext}')
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            f.save(filepath)

            report_name  = ''
            report_class = ''
            report_gender = ''
            report_school = ''
            grades       = {}
            report_text  = ''

            try:
                report_text = extract_text_from_file(filepath) or ''
            except Exception:
                report_text = ''

            try:
                grades = extract_grades_from_text(report_text) or {}
            except Exception:
                grades = {}

            try:
                if not looks_like_report_card_text(report_text, grades):
                    invalid_reports.append({
                        'slot_key': slot_key,
                        'slot': slot_label,
                        'file': f.filename,
                    })
            except Exception:
                invalid_reports.append({
                    'slot_key': slot_key,
                    'slot': slot_label,
                    'file': f.filename,
                })

            try:
                # ── Student information checks ────────────────────────────
                report_name = extract_name_from_text(report_text) or ''
                if report_name:
                    report_names.append({
                        'slot_key': slot_key,
                        'slot': slot_label,
                        'file': f.filename,
                        'name': report_name,
                    })
                if report_name and not names_match(student_name, report_name):
                    student_mismatches.append({
                        'slot_key': slot_key,
                        'slot':    slot_label,
                        'file':    f.filename,
                        'field':   'Name',
                        'entered': student_name,
                        'found':   report_name,
                    })

                report_gender = extract_gender_from_text(report_text) or ''
                if report_gender and not genders_match(gender, report_gender):
                    student_mismatches.append({
                        'slot_key': slot_key,
                        'slot':    slot_label,
                        'file':    f.filename,
                        'field':   'Gender',
                        'entered': 'Male' if gender == '1' else 'Female',
                        'found':   report_gender,
                    })

                report_school = extract_school_from_text(report_text) or ''
                if school_name and report_school and not schools_match(school_name, report_school):
                    student_mismatches.append({
                        'slot_key': slot_key,
                        'slot':    slot_label,
                        'file':    f.filename,
                        'field':   'School',
                        'entered': school_name,
                        'found':   report_school,
                    })
            except Exception:
                report_name = ''
                report_gender = ''
                report_school = ''

            try:
                # ── Wrong slot (wrong year) check ─────────────────────────
                report_class   = extract_class_from_text(report_text) or ''
                expected_class = SLOT_EXPECTED_CLASS.get(slot_key, '')
                if report_class and expected_class and report_class != expected_class:
                    slot_mismatches.append({
                        'slot_key':       slot_key,
                        'slot':           slot_label,
                        'file':           f.filename,
                        'expected_class': expected_class,
                        'found_class':    report_class,
                    })
            except Exception:
                report_class = ''

            all_grades.append(grades)
            uploaded_info.append({
                'label': slot_label, 'file': f.filename,
                'grades': grades, 'found': len(grades),
                'report_name':  report_name,
                'report_class': report_class,
                'report_gender': report_gender,
                'report_school': report_school,
            })

            try:
                os.remove(filepath)
            except Exception:
                pass
        else:
            uploaded_info.append({
                'label': slot_label, 'file': None, 'grades': {}, 'found': 0,
                'report_name': '', 'report_class': '',
                'report_gender': '', 'report_school': '',
            })

    # ── Block random PDFs/images that are not report cards ────────────────
    if invalid_reports:
        invalid_items = [
            f"{short_report_label(m['slot_key'], m['slot'])}: {m['file']}"
            for m in invalid_reports
        ]
        return redirect_with_modal({
            'type':     'danger',
            'title':    'Upload Report Cards Only',
            'subtitle': 'This file is not a report card',
            'body':     'Upload a real O-Level report card for:',
            'items':    invalid_items,
            'okText':   'Go Back & Upload Report',
        }, fallback_endpoint='upload')

    # ── Block if any report was uploaded into the wrong year slot ─────────
    if slot_mismatches:
        slot_items = [
            (f"{short_report_label(m['slot_key'], m['slot'])}: "
             f"{m['found_class']} report in {m['expected_class']} slot")
            for m in slot_mismatches
        ]
        return redirect_with_modal({
            'type':     'danger',
            'title':    'Wrong Slot',
            'subtitle': 'Put each report in its year slot',
            'body':     'Move these report cards:',
            'items':    slot_items,
            'okText':   'Go Back & Fix',
        }, fallback_endpoint='upload')

    # ── Block if the uploaded reports are for different students ──────────
    if len(report_names) >= 2 and report_names_conflict(report_names):
        name_items = [
            f"{short_report_label(m['slot_key'], m['slot'])}: {m['name']}"
            for m in report_names
        ]
        return redirect_with_modal({
            'type':     'danger',
            'title':    'Different Student Names',
            'subtitle': 'All reports must be for one student',
            'body':     'These reports show different names:',
            'items':    name_items,
            'okText':   'Go Back & Fix',
        }, fallback_endpoint='upload')

    # ── Block if any report belongs to a different student/school ─────────
    if student_mismatches:
        mismatch_items = [
            (f"{short_report_label(m['slot_key'], m['slot'])}: "
             f"{m['field']} is {m['found']}; entered {m['entered']}")
            for m in student_mismatches
        ]
        return redirect_with_modal({
            'type':     'danger',
            'title':    'Information Does Not Match',
            'subtitle': 'Check name, gender, or school',
            'body':     'Fix these details before continuing:',
            'items':    mismatch_items,
            'okText':   'Go Back & Fix',
        }, fallback_endpoint='upload')

    try:
        averaged      = average_grades(all_grades)
        averaged_pcts = average_percentages(all_grades)
    except Exception:
        averaged      = {}
        averaged_pcts = {}
    auto_count = len(averaged)

    return render_template('upload_review.html',
        student_name=student_name, gender=gender,
        guardian_email=guardian_email,
        averaged=averaged, averaged_pcts=averaged_pcts,
        uploaded_info=uploaded_info,
        subjects=SUBJECTS, grades=GRADES,
        auto_count=auto_count,
        wish_options=WISH_OPTIONS, career_options=CAREER_OPTIONS,
        talent_options=TALENT_OPTIONS, labor_options=LABOR_OPTIONS)


@app.route('/about')
def about():
    return render_template('about.html')


# ── Global error handlers — no ugly browser error pages ──────────────────
@app.errorhandler(500)
def internal_error(e):
    """Catch-all: redirect to upload page with a friendly modal instead of 500."""
    session['kiro_modal'] = {
        'type':     'danger',
        'title':    'Something Went Wrong',
        'subtitle': 'An unexpected error occurred while processing your files',
        'body':     (
            'This can happen if the uploaded file is corrupted, password-protected, '
            'or in an unsupported format. Please try again with a valid PDF or image file.'
        ),
        'okText':   'Go Back & Try Again',
    }
    return redirect(url_for('upload'))


@app.errorhandler(413)
def file_too_large(e):
    """File exceeded MAX_CONTENT_LENGTH (16 MB)."""
    session['kiro_modal'] = {
        'type':     'warning',
        'title':    'File Too Large',
        'subtitle': 'Maximum file size is 16 MB',
        'body':     'One or more of your files exceeds the 16 MB limit. Please upload a smaller file.',
        'okText':   'Go Back & Upload a Smaller File',
    }
    return redirect(url_for('upload'))


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', port=5000)
