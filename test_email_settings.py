import sys

from app import send_result_ready_email


def main():
    if len(sys.argv) != 2:
        print('Usage: python test_email_settings.py parent@example.com')
        return 1

    sample_results = [
        {'rank': 1, 'pathway': 'Mathematics & Sciences', 'probability': 82.5},
        {'rank': 2, 'pathway': 'TVET - Technology', 'probability': 10.4},
        {'rank': 3, 'pathway': 'Languages', 'probability': 4.6},
        {'rank': 4, 'pathway': 'Arts & Humanities', 'probability': 2.5},
    ]
    sample_grades = {
        'Math': 'A (85%)',
        'Physics': 'B (77%)',
        'Chemistry': 'C (72%)',
    }
    sample_choices = {
        'wish': 'Sciences',
        'career': 'Health & Medicine',
        'talent': 'Analytical / Problem-solving',
        'labor': 'Yes',
    }

    result = send_result_ready_email(
        sys.argv[1], 'Test Student',
        results=sample_results,
        grades=sample_grades,
        student_choices=sample_choices)
    if not result:
        print('No email address was provided.')
        return 1

    print(f"{result.get('status')}: {result.get('message')}")
    return 0 if result.get('status') == 'sent' else 1


if __name__ == '__main__':
    raise SystemExit(main())
