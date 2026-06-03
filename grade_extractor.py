"""
=============================================================================
 grade_extractor.py  —  Grade Extraction from Rwandan O-Level Report Cards
=============================================================================
"""

import re, os
import pdfplumber
import pytesseract
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

GRADE_VALUES = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'S': 2, 'F': 1}

SUBJECTS = ['Math', 'Physics', 'Chemistry', 'Biology',
            'Geography', 'History', 'Entrepreneurship', 'English', 'Kinyarwanda',
            'French', 'Kiswahili']

SUBJECT_ALIASES = {
    'Math':            ['mathematics', 'mathematiques', 'maths', 'math'],
    'Physics':         ['physics', 'physique'],
    'Chemistry':       ['chemistry', 'chimie'],
    'Biology':         ['biology and health sciences', 'biology and health',
                        'biology', 'biologie'],
    'Geography':       ['geography and environment', 'geography', 'geographie'],
    'History':         ['history and citizenship', 'history', 'histoire'],
    'Entrepreneurship':['entrepreneurship', 'entrepreneuriat',
                        'entreprenuership', 'entrepren'],
    'English':         ['english', 'anglais'],
    'Kinyarwanda':     ['ikinyarwanda', 'kinyarwanda', 'kinyarw'],
    'French':          ['french', 'francais', 'franc'],
    'Kiswahili':       ['kiswahili', 'swahili', 'kiswah'],
}

# Fallback midpoints used ONLY when the real % cannot be read from the PDF
_GRADE_MIDPOINTS = {'A': 85.0, 'B': 77.0, 'C': 72.0, 'D': 67.0,
                    'E': 62.0, 'S': 55.0, 'F': 40.0}

# ── helpers ───────────────────────────────────────────────────────────────

def _pct_to_letter(pct: float) -> str:
    if pct >= 80: return 'A'
    if pct >= 75: return 'B'
    if pct >= 70: return 'C'
    if pct >= 65: return 'D'
    if pct >= 60: return 'E'
    if pct >= 50: return 'S'
    return 'F'


SKIP_LINES = [
    r'^subject', r'^maximum', r'^weight', r'^conduct',
    r'^all sub', r'^term\s+\d', r'^annual', r'^total\b',
    r'^position', r'^percentage', r'^grading', r'^final\b',
    r'^eu\b', r'^et\b', r'^tot\b', r'^gr\b', r'^core sub',
    r'^comment', r'^parent', r'^class\s?teacher', r'^ts\b',
]

def _is_skip(line):
    lo = line.lower().strip()
    return any(re.match(p, lo) for p in SKIP_LINES)

def _detect_subject(line):
    if _is_skip(line):
        return None
    lo = line.lower().strip()
    best_key, best_len = None, 0
    for key, aliases in SUBJECT_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if lo.startswith(alias) or (len(lo) <= 35 and alias in lo):
                if len(alias) > best_len:
                    best_len = len(alias)
                    best_key = key
    return best_key

def _last_grade(line):
    hits = re.findall(r'(?<![A-Za-z])([A-FS])(?![A-Za-z])', line)
    for h in reversed(hits):
        if h.upper() in GRADE_VALUES:
            return h.upper()
    return None

def _last_pct(line: str):
    """
    Extract Annual-Total percentage from a subject row.

    CAMIS PDFs do NOT put a '%' sign after data values — '%' only appears
    in the column header row.  We use three strategies:

    1. Explicit 'xx.x%' token  (some older/non-CAMIS formats).
    2. All 'decimal<space>letter' pairs e.g. '83.3 A' — return the last
       valid one.  Handles OCR merging the next subject onto the same line.
    3. Last bare decimal in range 30-100 as a final fallback.
    """
    # 1. explicit %
    for p in reversed(re.findall(r'(\d{1,3}(?:\.\d+)?)\s*%', line)):
        try:
            v = float(p)
            if 0.0 <= v <= 100.0:
                return v
        except ValueError:
            pass

    # 2. bare 'decimal letter' pairs
    for m in reversed(list(re.finditer(r'(\d{2,3}\.\d)\s+([A-FS])(?=\s|$)', line))):
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 100.0:
                return v
        except ValueError:
            pass

    # 3. last bare decimal 30-100
    for n in reversed(re.findall(r'\b(\d{2,3}\.\d+)\b', line)):
        try:
            v = float(n)
            if 30.0 <= v <= 100.0:
                return v
        except ValueError:
            pass

    return None

# ── core parser ───────────────────────────────────────────────────────────

def _parse_text_with_pcts(text: str) -> dict:
    """
    Parse report card text.
    Returns { subject: {'grade': 'A', 'pct': 83.3} }
    pct is the real Annual-Total percentage, or None if not readable.
    """
    results = {}
    lines   = text.split('\n')

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        sub = _detect_subject(line)
        if not sub or sub in results:
            continue
        merged = line
        for j in range(1, 3):
            if i + j < len(lines):
                nxt = lines[i + j].strip()
                if nxt and not _detect_subject(nxt):
                    merged += ' ' + nxt
        pct = _last_pct(merged)
        g   = _last_grade(merged)
        if not g and pct is not None:
            g = _pct_to_letter(pct)
        if g:
            results[sub] = {'grade': g, 'pct': pct}

    if len(results) < 9:
        lo = text.lower()
        for key, aliases in SUBJECT_ALIASES.items():
            if key in results:
                continue
            for alias in sorted(aliases, key=len, reverse=True):
                idx = lo.find(alias)
                if idx == -1:
                    continue
                snip = text[idx: idx + 300]
                pct  = _last_pct(snip)
                g    = _last_grade(snip) or \
                       (_pct_to_letter(pct) if pct is not None else None)
                if g:
                    results[key] = {'grade': g, 'pct': pct}
                    break

    return results

def _parse_text(text: str) -> dict:
    return {s: v['grade'] for s, v in _parse_text_with_pcts(text).items()}

# ── image / pdf reading ───────────────────────────────────────────────────

def _preprocess(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w < 2400:
        s = 2400 / w
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    grey = img.convert('L')
    if HAVE_CV2:
        arr = np.array(grey)
        arr = cv2.adaptiveThreshold(arr, 255,
              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        arr = cv2.dilate(arr, np.ones((2, 2), np.uint8), iterations=1)
        return Image.fromarray(arr)
    grey = ImageEnhance.Contrast(grey).enhance(3.0)
    grey = ImageEnhance.Sharpness(grey).enhance(2.5)
    return grey.filter(ImageFilter.MedianFilter(3))

def _subject_score(text: str) -> int:
    lo = text.lower()
    return sum(1 for aliases in SUBJECT_ALIASES.values()
               if any(a in lo for a in aliases))

def _best_ocr(img: Image.Image) -> str:
    best, best_s = '', -1
    for angle in [0, 90, 270, 180]:
        rim = img.rotate(angle, expand=True) if angle else img
        for psm in [6, 4]:
            try:
                pre = _preprocess(rim)
                t   = pytesseract.image_to_string(pre,
                          config=f'--psm {psm} --oem 3 -l eng')
                s   = _subject_score(t) * 10 + len(t) // 200
                if s > best_s:
                    best_s, best = s, t
            except Exception:
                continue
    return best

def _read_file_text(filepath: str) -> str:
    ext  = os.path.splitext(filepath)[1].lower()
    text = ''
    if ext == '.pdf':
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + '\n'
        except Exception:
            pass
        if not text.strip():
            try:
                for img in convert_from_path(filepath, dpi=250):
                    text += _best_ocr(img) + '\n'
            except Exception:
                pass
    elif ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'):
        try:
            text = _best_ocr(Image.open(filepath))
        except Exception:
            pass
    return text

# ══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def extract_text_from_file(filepath: str) -> str:
    """Read searchable/OCR text from a PDF or image upload."""
    return _read_file_text(filepath)


def extract_grades_from_text(text: str) -> dict:
    """Returns {subject: {'grade':'A','pct':83.3}} from already-read text."""
    return _parse_text_with_pcts(text) if text and text.strip() else {}


def extract_grades_from_file(filepath: str) -> dict:
    """Returns {subject: {'grade':'A','pct':83.3}} — preserves real Annual-Total %."""
    text = _read_file_text(filepath)
    return _parse_text_with_pcts(text) if text.strip() else {}


def average_percentages(grades_list: list) -> dict:
    """
    Average the real Annual-Total percentages from 4 report cards.
    grades_list: list of dicts from extract_grades_from_file()
    Returns {subject: averaged_percentage}
    """
    totals, counts = {}, {}
    for report in grades_list:
        if not report:
            continue
        for sub, value in report.items():
            if isinstance(value, dict):
                raw = value.get('pct')
                g   = value.get('grade', '')
                if raw is not None:
                    try:
                        pct = float(raw)
                        pct = pct if 0.0 <= pct <= 100.0 else None
                    except (ValueError, TypeError):
                        pct = None
                else:
                    pct = None
                if pct is None:
                    pct = _GRADE_MIDPOINTS.get(g.upper() if g else '')
            elif isinstance(value, (int, float)):
                pct = float(value)
            elif isinstance(value, str):
                pct = _GRADE_MIDPOINTS.get(value.upper())
            else:
                continue
            if pct is not None:
                totals[sub] = totals.get(sub, 0.0) + pct
                counts[sub] = counts.get(sub, 0) + 1
    return {sub: round(totals[sub] / counts[sub], 1) for sub in totals}


def average_grades(grades_list: list) -> dict:
    """
    Returns averaged letter grades.
    Accepts both {subject:'A'} and {subject:{'grade':'A','pct':83.3}} formats.
    """
    totals, counts = {}, {}
    for report in grades_list:
        if not report:
            continue
        for sub, value in report.items():
            grade = value['grade'] if isinstance(value, dict) else value
            val   = GRADE_VALUES.get(grade)
            if val:
                totals[sub] = totals.get(sub, 0) + val
                counts[sub] = counts.get(sub, 0) + 1
    reverse = {6: 'A', 5: 'B', 4: 'C', 3: 'D', 2: 'E', 1: 'F'}
    return {
        sub: reverse[max(1, min(6, round(totals[sub] / counts[sub])))]
        for sub in totals
    }


def extract_class_from_text(text: str) -> str:
    m = re.search(r'class\s*[:\-]\s*S\s*([123])', text, re.IGNORECASE)
    if m: return f'S{m.group(1)}'
    m = re.search(r'classe\s*[:\-]\s*S\s*([123])', text, re.IGNORECASE)
    if m: return f'S{m.group(1)}'
    return ''

def extract_class_from_file(filepath: str) -> str:
    ext, text = os.path.splitext(filepath)[1].lower(), ''
    if ext == '.pdf':
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages[:2]:
                    t = page.extract_text()
                    if t: text += t + '\n'
        except Exception: pass
        if not text.strip():
            try:
                for img in convert_from_path(filepath, dpi=200, first_page=1, last_page=1):
                    text += _best_ocr(img) + '\n'
            except Exception: pass
    elif ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'):
        try: text = _best_ocr(Image.open(filepath))
        except Exception: pass
    return extract_class_from_text(text)

def extract_school_from_text(text: str) -> str:
    for line in text.splitlines()[:60]:
        if re.search(r'\bschool\s+code\b', line, re.IGNORECASE):
            continue
        m = re.search(r'\bschool\s*[:\-]\s*([^\n\r]{2,90})', line, re.IGNORECASE)
        if not m:
            m = re.search(r'\becole\s*[:\-]\s*([^\n\r]{2,90})', line, re.IGNORECASE)
        if m:
            school = m.group(1).strip()
            school = re.split(
                r'\s+(?:School Code|E-mail|Email|Phone|Names?|Academic Year|Registration|Level|Class|District)\b',
                school,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(' :-')
            if len(school) >= 2:
                return re.sub(r'\s+', ' ', school).upper()
    return ''


def extract_school_from_file(filepath: str) -> str:
    return extract_school_from_text(_read_file_text(filepath))


def _normalise_gender(raw: str) -> str:
    raw = (raw or '').strip().lower()
    if raw in ('m', 'male', 'masculin', 'masculine', 'boy', 'garcon', 'garçon'):
        return 'Male'
    if raw in ('f', 'female', 'feminin', 'feminine', 'féminin', 'girl', 'fille'):
        return 'Female'
    return ''


def extract_gender_from_text(text: str) -> str:
    for line in text.splitlines()[:80]:
        m = re.search(r'\b(?:gender|sex|sexe)\s*[:\-]\s*([A-Za-zéÉçÇ]+)', line, re.IGNORECASE)
        if m:
            gender = _normalise_gender(m.group(1))
            if gender:
                return gender
    return ''


def extract_gender_from_file(filepath: str) -> str:
    return extract_gender_from_text(_read_file_text(filepath))


def extract_name_from_text(text: str) -> str:
    patterns = [
        r'names?\s*[:\-]\s*([A-Z][A-Za-z\s\-\']{2,60})',
        r'student\s*[:\-]\s*([A-Z][A-Za-z\s\-\']{2,60})',
        r'pupil\s*[:\-]\s*([A-Z][A-Za-z\s\-\']{2,60})',
        r'nom\s*[:\-]\s*([A-Z][A-Za-z\s\-\']{2,60})',
        r'élève\s*[:\-]\s*([A-Z][A-Za-z\s\-\']{2,60})',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            name = re.split(r'\n|Registration|Class|Academic|Level|School',
                            name, maxsplit=1)[0].strip()
            if len(name) >= 3: return name.upper()
    return ''

def extract_name_from_file(filepath: str) -> str:
    ext, text = os.path.splitext(filepath)[1].lower(), ''
    if ext == '.pdf':
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages[:2]:
                    t = page.extract_text()
                    if t: text += t + '\n'
        except Exception: pass
        if not text.strip():
            try:
                for img in convert_from_path(filepath, dpi=200, first_page=1, last_page=1):
                    text += _best_ocr(img) + '\n'
            except Exception: pass
    elif ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'):
        try: text = _best_ocr(Image.open(filepath))
        except Exception: pass
    return extract_name_from_text(text)

def _name_tokens(name: str) -> set:
    return {t.upper() for t in re.split(r'[\s\-]+', name.strip()) if len(t) >= 2}

def names_match(entered: str, extracted: str, threshold: float = 1.0) -> bool:
    if not entered or not extracted: return True
    et = _name_tokens(entered)
    xt = _name_tokens(extracted)
    if not xt: return True
    return len(et & xt) / len(xt) >= threshold


def genders_match(entered: str, extracted: str) -> bool:
    if not extracted:
        return True
    entered_label = _normalise_gender('Male' if str(entered) == '1' else
                                      'Female' if str(entered) == '0' else entered)
    extracted_label = _normalise_gender(extracted)
    if not entered_label or not extracted_label:
        return True
    return entered_label == extracted_label


def _school_tokens(name: str) -> set:
    name = (name or '').upper()
    name = re.sub(r'\bG\s*\.?\s*S\b', 'GS', name)
    name = re.sub(r'\bGROUPE\s+SCOLAIRE\b', 'GS', name)
    name = re.sub(r'[^A-Z0-9]+', ' ', name)
    stop = {
        'GS', 'SCHOOL', 'ECOLE', 'COLLEGE', 'SECONDARY', 'PRIMARY',
        'TSS', 'TVET', 'LTD', 'RCA', 'REB', 'MINEDUC',
    }
    return {t for t in name.split() if len(t) > 1 and t not in stop}


def schools_match(entered: str, extracted: str, threshold: float = 0.6) -> bool:
    if not entered or not extracted:
        return True
    et = _school_tokens(entered)
    xt = _school_tokens(extracted)
    if not et or not xt:
        return True
    overlap = et & xt
    return bool(overlap) and (len(overlap) / min(len(et), len(xt)) >= threshold)


def looks_like_report_card_text(text: str, grades: dict = None) -> bool:
    """Reject random PDFs/images before sending users to grade review."""
    if not text or not text.strip():
        return False
    grade_count = len(grades) if grades is not None else len(extract_grades_from_text(text))
    subject_count = _subject_score(text)
    has_report_title = re.search(r'\b(?:student\s+)?report\s+card\b', text, re.IGNORECASE)
    support_patterns = [
        r'\bacademic\s+year\b',
        r'\bannual\s+total\b',
        r'\bfinal\s+grade\b',
        r'\bgrading\s+scale\b',
        r'\bregistration\s+id\b',
        r'\bnames?\s*:',
        r'\bclass\s*:',
        r'\bschool\s*:',
    ]
    support_count = sum(1 for pat in support_patterns if re.search(pat, text, re.IGNORECASE))
    if has_report_title and (support_count >= 2 or subject_count >= 3 or grade_count >= 3):
        return True
    return support_count >= 2 and (subject_count >= 5 or grade_count >= 3)
